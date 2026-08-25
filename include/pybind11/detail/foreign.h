/*
    pybind11/detail/foreign.h: Interoperability with other binding frameworks

    Copyright (c) 2025 Hudson River Trading <opensource@hudson-trading.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include <pybind11/contrib/pymetabind.h>

#include "common.h"
#include "internals.h"
#include "type_caster_base.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

// NB: In pymetabind callbacks you can use foreign_internals_local_cache()
// because the callbacks are always from the same DSO that created the
// foreign_internals object. This is a bit faster and also has the advantage
// of working correctly during interpreter finalization. In other functions,
// which could be called in a different DSO, use get_foreign_internals() which
// will access the pointer via our local_internals structure.

// pybind11 exception translator that tries all known foreign ones;
// this will be registered in the pybind11 list of exception translators,
// in the 2nd position from the end (last position is for the fallback translator)
PYBIND11_NOINLINE void foreign_exception_translator(std::exception_ptr p) {
    auto *foreign_internals = get_foreign_internals();
    if (foreign_internals) { // might be null during interpreter finalization
        for (pymb_framework *fw : foreign_internals->exc_frameworks) {
            if (fw->translate_exception(&p) != 0) {
                return;
            }
        }
    }
    std::rethrow_exception(p);
}

// When learning about a new foreign type, should we automatically use it?
inline bool should_autoimport_foreign(foreign_internals &foreign_internals,
                                      pymb_binding *binding) {
    return foreign_internals.autoimport_for_anyone
           && binding->framework->abi_lang == pymb_abi_lang_cpp
           && binding->framework->abi_extra == foreign_internals.self->abi_extra;
}

// Determine whether a pybind11 type is module-local from a different module
inline bool is_local_to_other_module(type_info *ti) {
    return ti->module_local_load != nullptr
           && ti->module_local_load != &type_caster_generic::local_load;
}

// Add the given `binding` to our type maps so that we can use it to satisfy
// from- and to-Python requests for the given C++ type
inline void import_foreign_binding(pymb_binding *binding,
                                   const std::type_info *cpptype,
                                   bool manual) noexcept {
    // Caller must hold the internals lock
    auto &foreign_internals = *get_foreign_internals();
    foreign_internals.imported_any = true;
    auto &lst = foreign_internals.bindings[*cpptype];
    auto *pos = lst.find(binding);
    if (!pos) {
        ++foreign_internals.bindings_update_count;
        lst.push_back(binding);
        pos = lst.data() + lst.size() - 1;
    }
    if (manual && pos != lst.data()) {
        // Move manual imports to the front of the list so they will
        // be preferred for future conversions.
        ++foreign_internals.bindings_update_count;
        std::move_backward(lst.data(), pos, pos + 1);
        lst[0] = binding;
    }
}

// ----------------------------------------------------------------------
// pymetabind callback functions that other frameworks will use to operate
// on our objects or tell us about theirs
// ----------------------------------------------------------------------

// NB: for all pymetabind bindings provided by pybind11, pymb_binding::context
// is either the detail::type_info* (for a py::class_ or py::enum_) or nullptr
// (for a py::native_enum).

inline void *foreign_cb_from_python(pymb_binding *binding,
                                    PyObject *pyobj,
                                    uint8_t convert,
                                    void (*keep_referenced)(void *ctx, PyObject *obj),
                                    void *keep_referenced_ctx) noexcept {
    if (binding->context == nullptr) {
        // This is a native enum type. We can only return a pointer to the C++
        // enum if we're able to allocate a temporary, since there aren't any
        // bytes inside the Python enum instance that can validly be interpreted
        // as a C++ enum object.
        handle pytype(reinterpret_cast<PyObject *>(binding->pytype));
        if (!keep_referenced || !isinstance(pyobj, pytype)) {
            return nullptr;
        }
        try {
            // Get the pybind11 enum record for this enum
            auto cap
                = reinterpret_borrow<capsule>(pytype.attr(native_enum_record::attribute_name()));
            auto *info = cap.get_pointer<native_enum_record>();

            // The C++ integer enum value is stored as the `value` attribute
            // of the Python enum object. Obtain it as a py::int_ object, then
            // extract the underlying integer. If negative, cast to unsigned
            // in order to reduce duplication of code paths; this preserves the
            // original bit pattern on any reasonable implementation.
            auto value = handle(pyobj).attr("value");
            uint64_t ival = 0;
            if (info->is_signed && handle(value) < int_(0)) {
                ival = static_cast<uint64_t>(cast<int64_t>(value));
            } else {
                ival = cast<uint64_t>(value);
            }

            // Copy from `ival` the correct number of low-order bytes for the
            // actual enum type. The bytes we need are the first ones on
            // little-endian platforms and the last on big-endian.
            // We could also switch between 8 codepaths for signed/unsigned
            // 1/2/4/8-byte, but doing it this way is less bloaty.
            bytes holder{reinterpret_cast<const char *>(&ival)
                             + PYBIND11_BIG_ENDIAN * size_t(8 - info->size_bytes),
                         info->size_bytes};

            // Ask the caller to keep our `holder` alive until they're done using
            // the pointer we gave them, and return the pointer to its contents.
            keep_referenced(keep_referenced_ctx, holder.ptr());
            return PyBytes_AsString(holder.ptr());
        } catch (error_already_set &exc) {
            // pymetabind doesn't provide any way to return full exception info
            // for from-Python conversions, so fall back on writing it to stderr
            exc.discard_as_unraisable("Error converting native enum from Python");
            return nullptr;
        }
    }

    // If it's not a native enum, it must be a py::class_ type (which includes
    // old-style py::enum_).

#if defined(PYBIND11_HAS_OPTIONAL)
    using maybe_life_support = std::optional<loader_life_support>;
#else
    // No-frills duplicate of partial functionality of std::optional, for
    // backward compatibility with compilers that don't support it.
    struct maybe_life_support {
        union {
            loader_life_support supp;
        };
        bool engaged = false;

        maybe_life_support() {}
        maybe_life_support(maybe_life_support &) = delete;
        loader_life_support *operator->() { return &supp; }
        void emplace() {
            new (&supp) loader_life_support();
            engaged = true;
        }
        ~maybe_life_support() {
            if (engaged) {
                supp.~loader_life_support();
            }
        }
    };
#endif

    // Create a loader_life_support only if we will be able to hand off
    // (to our caller) the ownership of the references added to it
    maybe_life_support holder;
    if (keep_referenced) {
        holder.emplace();
    }
    type_caster_generic caster{static_cast<const type_info *>(binding->context)};
    void *ret = nullptr;
    try {
        if (caster.load_impl<type_caster_generic>(pyobj,
                                                  convert != 0,
                                                  /* foreign_ok */ false)) {
            ret = caster.value;
        }
    } catch (...) {
        // pymetabind doesn't provide any way to return full exception info
        // for from-Python conversions, so fall back on writing it to stderr
        translate_exception(std::current_exception());
        PyErr_WriteUnraisable(pyobj);
    }
    if (keep_referenced) {
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        for (PyObject *item : holder->list_patients()) {
            keep_referenced(keep_referenced_ctx, item);
        }
    }
    return ret;
}

// This wraps the call to type_info::init_instance() in some cases when casting
// a pybind11-bound object to Python on behalf of a foreign framework. It
// inhibits registration of the new instance so that foreign_cb_keep_alive()
// can fix up the holder before other threads start using the new instance.
inline void init_instance_unregistered(instance *inst, const void *holder) {
    assert(holder == nullptr && !inst->owned);
    (void) holder; // avoid unused warning if compiled without asserts
    value_and_holder v_h = *values_and_holders(inst).begin();

    // Pretend it's already registered so that init_instance doesn't try again
    v_h.set_instance_registered(true);

    // Undo our shenanigans even if init_instance raises an exception
    struct guard {
        value_and_holder &v_h;
        ~guard() noexcept { v_h.set_instance_registered(false); }
    } guard{v_h};
    v_h.type->init_instance(inst, nullptr);
}

inline PyObject *foreign_cb_to_python(pymb_binding *binding,
                                      void *cobj,
                                      enum pymb_rv_policy rvp_,
                                      pymb_to_python_feedback *feedback) noexcept {
    feedback->relocate = 0; // we don't support relocation
    feedback->is_new = 0;   // unless overridden below

    if (cobj == nullptr) {
        // Converting null to Python produces a None object
        return none().release().ptr();
    }

    if (binding->context == nullptr) {
        // Native enum type
        try {
            // Get the pybind11 enum record for this enum type, so we can
            // determine its size and signedness
            handle pytype(reinterpret_cast<PyObject *>(binding->pytype));
            auto cap
                = reinterpret_borrow<capsule>(pytype.attr(native_enum_record::attribute_name()));
            auto *info = cap.get_pointer<native_enum_record>();

            // Get the underlying integer value of the C++ enum object
            // we're converting to Python
            uint64_t key = 0;
            switch (info->size_bytes) {
                case 1:
                    key = *reinterpret_cast<uint8_t *>(cobj);
                    break;
                case 2:
                    key = *reinterpret_cast<uint16_t *>(cobj);
                    break;
                case 4:
                    key = *reinterpret_cast<uint32_t *>(cobj);
                    break;
                case 8:
                    key = *reinterpret_cast<uint64_t *>(cobj);
                    break;
                default:
                    return nullptr;
            }
            if (rvp_ == pymb_rv_policy_take_ownership) {
                // Rare case: returning a heap-allocated enum; the Python
                // enum object can't actually carry ownership, so delete it
                // immediately now that we've extracted the contents
                ::operator delete(cobj);
            }
            if (info->is_signed) {
                // Reinterpret the unsigned value we extracted above as signed
                auto ikey = static_cast<int64_t>(key);
                if (info->size_bytes < 8) {
                    // If we extracted less than 8 bytes, we need to sign
                    // extend (fill the higher-order bits of the int64 with
                    // the sign bit of the smaller value we extracted); this
                    // is a standard idiom for that
                    ikey <<= (64 - (info->size_bytes * 8));
                    ikey >>= (64 - (info->size_bytes * 8));
                }
                return pytype(ikey).release().ptr();
            }
            return pytype(key).release().ptr();
        } catch (error_already_set &exc) {
            exc.restore();
            return nullptr;
        }
    }

    // Otherwise this is a py::class_ or py::enum_ binding

    const auto *ti = static_cast<const type_info *>(binding->context);
    return_value_policy rvp = return_value_policy::automatic;
    bool inhibit_registration = false;

    // Convert pymetabind RVP to pybind11 RVP
    switch (rvp_) {
        case pymb_rv_policy_take_ownership:
        case pymb_rv_policy_copy:
        case pymb_rv_policy_move:
        case pymb_rv_policy_reference:
            // These have the same numeric values and semantics as our own policies
            rvp = static_cast<return_value_policy>(rvp_);
            break;
        case pymb_rv_policy_share_ownership:
            // Treat as `reference` for the initial creation; we'll fix it up
            // later
            rvp = return_value_policy::reference;
            inhibit_registration = true;
            break;
        case pymb_rv_policy_none:
            break;
    }
    if (rvp == return_value_policy::automatic) {
        // Specified rvp was none, or was something unrecognized so we should
        // be conservative and treat it like none.
        return find_registered_python_instance(cobj, ti).ptr();
    }

    copy_or_move_ctor copy_ctor = nullptr, move_ctor = nullptr;
    if (rvp == return_value_policy::copy || rvp == return_value_policy::move) {
        // If we're making a copy or a move, we need to fetch the copy
        // and move constructors. Ideally they would be stored in the type_info
        // but for now we have a parallel map in order to maintain ABI compat.
        // If they're not present (such as because this pytype was created by
        // a version of pybind11 that didn't know about pymetabind) we'll
        // leave them as null and accept a failure of cast() below.
        with_internals([&](internals &) {
            auto &foreign_internals = *foreign_internals_local_cache();
            auto it = foreign_internals.copy_move_ctors.find(*ti->cpptype);
            if (it != foreign_internals.copy_move_ctors.end()) {
                std::tie(copy_ctor, move_ctor) = it->second;
            }
        });
    }

    try {
        cast_sources srcs{cobj, ti};
        if (inhibit_registration) {
            srcs.init_instance = init_instance_unregistered;
        }
        handle ret = type_caster_generic::cast(srcs, rvp, {}, copy_ctor, move_ctor);
        feedback->is_new = static_cast<uint8_t>(srcs.is_new);
        return ret.ptr();
    } catch (...) {
        translate_exception(std::current_exception());
        return nullptr;
    }
}

inline int foreign_cb_keep_alive(PyObject *nurse,
                                 pymb_keep_alive_type type,
                                 void *payload,
                                 void (*cb)(void *)) noexcept {
    try {
        // This do-while loop only executes one iteration, and exists purely
        // to reduce the nesting level by allowing use of `break` statements.
        // Within it, we're trying to represent this keep-alive by setting a
        // shared_ptr on `nurse` with an appropriate deleter, which is
        // only possible for an object that was freshly created using
        // pymb_rv_policy_share_ownership. After the do-while we'll handle
        // the more common keep alive case. The special shared_ptr cases here
        // are essential to allow a shared_ptr<T> returned from a foreign-bound
        // function to be acceptable as a shared_ptr<T> argument to a
        // pybind11-bound function.
        do {
            if (!is_uniquely_referenced(nurse)) {
                break; // someone else might hold a reference to this object
            }
            values_and_holders vhs{nurse};
            if (vhs.size() != 1) {
                break; // instance of a pytype that inherits multiple cpptypes
            }
            value_and_holder v_h = *vhs.begin();
            if (v_h.instance_registered()) {
                break; // someone else might concurrently obtain a reference
            }
            // After this point, no early break allowed -- we're committing to
            // register the instance regardless of whether we successfully set
            // the keep_alive via shared_ptr or not.
            bool can_set_shared_ptr = (v_h.type->holder_enum_v == holder_enum_t::std_shared_ptr
                                       && !v_h.holder_constructed());
            bool can_set_smart_holder = (v_h.type->holder_enum_v == holder_enum_t::smart_holder
                                         && v_h.holder_constructed() && !v_h.inst->owned);
            bool success = false;
            if (can_set_shared_ptr || can_set_smart_holder) {
                // Create a shared_ptr that carries the requested ownership.
                std::shared_ptr<void> result;
                switch (type) {
                    case pymb_keep_alive_callback:
                        result = std::shared_ptr<void>(payload, cb);
                        break;
                    case pymb_keep_alive_pyobject: {
                        auto *patient = static_cast<PyObject *>(payload);
                        Py_INCREF(patient);
                        result = std::shared_ptr<PyObject>(patient, Py_DecRef);
                        break;
                    }
                    case pymb_keep_alive_cpp_shared_ptr_void: {
                        auto *given = static_cast<std::shared_ptr<void> *>(payload);
                        result = std::move(*given);
                        break;
                    }
                }
                if (result) {
                    // Use the aliasing constructor so that result.get() returns the
                    // right thing, despite its deleter receiving a possibly-unrelated
                    // `payload`. NB: this constructor accepts an rvalue reference only
                    // in C++20, so suppress a lint for the sake of earlier versions.
                    // NOLINTNEXTLINE(performance-move-const-arg)
                    result = std::shared_ptr<void>(std::move(result), v_h.value_ptr());
                    if (can_set_shared_ptr) {
                        new (std::addressof(v_h.holder<std::shared_ptr<void>>()))
                            std::shared_ptr<void>(std::move(result));
                        v_h.set_holder_constructed();
                    } else {
                        assert(can_set_smart_holder);
                        auto &h = v_h.holder<smart_holder>();
                        // Since inst->owned was false and we did not pass a
                        // holder, it was probably created by
                        // `smart_holder::from_raw_ptr_unowned`. (If the pointee
                        // implements `enable_shared_from_this`, `init_instance`
                        // would have instead already done something like the
                        // below, but it works fine in that case too.)
                        // Undo the effects of `from_raw_ptr_unowned` and set up
                        // the holder as if by `smart_holder::from_shared_ptr`.
                        h.vptr = std::move(result);
                        h.vptr_is_using_noop_deleter = false;
                        h.vptr_is_using_std_default_delete = false;
                        h.vptr_is_external_shared_ptr = true;
                        h.is_populated = true;
                    }
                    success = true;
                }
            }
            register_instance(v_h.inst, v_h.value_ptr(), v_h.type);
            v_h.set_instance_registered(true);
            if (success) {
                return 1;
            }
        } while (false);

        // Normal keep-alive logic for all instances except those newly
        // created with pymb_rv_policy_share_ownership
        switch (type) {
            case pymb_keep_alive_callback: {
                capsule patient{payload, cb};
                keep_alive_impl(nurse, patient);
                return 1;
            }
            case pymb_keep_alive_pyobject:
                keep_alive_impl(nurse, static_cast<PyObject *>(payload));
                return 1;
            case pymb_keep_alive_cpp_shared_ptr_void: {
                auto *given = static_cast<std::shared_ptr<void> *>(payload);
                capsule patient{new std::shared_ptr<void>{std::move(*given)},
                                +[](void *p) { delete static_cast<std::shared_ptr<void> *>(p); }};
                keep_alive_impl(nurse, patient);
                return 1;
            }
        }
        return 0;
    } catch (...) {
        translate_exception(std::current_exception());
        PyErr_WriteUnraisable(nurse);
        return 0;
    }
}

inline int foreign_cb_translate_exception(void *eptr) noexcept {
    return with_exception_translators(
        [&](std::forward_list<ExceptionTranslator> &exception_translators,
            std::forward_list<ExceptionTranslator> & /*local_exception_translators*/) {
            // Ignore local exception translators. We're being called to translate
            // an exception that was raised from a different framework, thus a
            // different extension module, so nothing local to us will apply.
            // Try global translators, except the last one or two.
            std::exception_ptr &e = *static_cast<std::exception_ptr *>(eptr);
            auto it = exception_translators.begin();
            // We will iterate `it` and `leader` in lockstep until `leader`
            // reaches the end of the list, so however far past `it` we advance
            // `leader` is the number of trailing translators we won't call.
            auto leader = it;
            // - The last one is the default translator. It translates
            //   standard exceptions, which we should leave up to the
            //   framework that bound a function.
            ++leader;
            // - If we've installed the foreign_exception_translator hook
            //   (for pybind11 functions to be able to translate other
            //   frameworks' exceptions), it's the second-last one and should
            //   be skipped too. We don't want mutual recursion between
            //   different frameworks' translators.
            if (!foreign_internals_local_cache()->exc_frameworks.empty()) {
                ++leader;
            }

            for (; leader != exception_translators.end(); ++it, ++leader) {
                try {
                    (*it)(e);
                    return 1;
                } catch (...) {
                    e = std::current_exception();
                }
            }

            // Try the part of the default translator that is pybind11-specific
            try {
                std::rethrow_exception(e);
            } catch (error_already_set &err) {
                handle_nested_exception(err, e);
                err.restore();
                return 1;
            } catch (const builtin_exception &err) {
                // Could not use template since it's an abstract class.
                if (const auto *nep
                    = dynamic_cast<const std::nested_exception *>(std::addressof(err))) {
                    handle_nested_exception(*nep, e);
                }
                err.set_error();
                return 1;
            } catch (...) {
                e = std::current_exception();
            }
            return 0;
        });
}

inline void foreign_cb_remove_local_binding(pymb_binding *binding) noexcept {
    with_internals([&](internals &) {
        auto &foreign_internals = *foreign_internals_local_cache();
        const auto *cpptype = (const std::type_info *) binding->native_type;
        auto it = foreign_internals.bindings.find(*cpptype);
        if (it != foreign_internals.bindings.end() && it->second.erase_one(binding)) {
            ++foreign_internals.bindings_update_count;
            if (it->second.empty()) {
                foreign_internals.bindings.erase(it);
            }
        }
    });
}

inline void foreign_cb_free_local_binding(pymb_binding *binding) noexcept {
    free(const_cast<char *>(binding->source_name));
    delete binding;
}

inline void foreign_cb_add_foreign_binding(pymb_binding *binding) noexcept {
    with_internals([&](internals &) {
        auto &foreign_internals = *foreign_internals_local_cache();
        if (should_autoimport_foreign(foreign_internals, binding)) {
            import_foreign_binding(binding, (const std::type_info *) binding->native_type, false);
        }
    });
}

inline void foreign_cb_remove_foreign_binding(pymb_binding *binding) noexcept {
    with_internals([&](internals &) {
        auto &foreign_internals = *foreign_internals_local_cache();
        auto remove_from_type = [&](const std::type_info *type) {
            auto it = foreign_internals.bindings.find(*type);
            if (it != foreign_internals.bindings.end() && it->second.erase_one(binding)) {
                ++foreign_internals.bindings_update_count;
                if (it->second.empty()) {
                    foreign_internals.bindings.erase(it);
                }
            }
        };
        bool should_remove_auto = should_autoimport_foreign(foreign_internals, binding);
        auto it = foreign_internals.manual_imports.find(binding);
        if (it != foreign_internals.manual_imports.end()) {
            remove_from_type(it->second);
            should_remove_auto &= (it->second != binding->native_type);
            foreign_internals.manual_imports.erase(it);
        }
        if (should_remove_auto) {
            remove_from_type((const std::type_info *) binding->native_type);
        }
    });
}

inline void foreign_cb_add_foreign_framework(pymb_framework *framework) noexcept {
    if (framework->translate_exception) {
        with_exception_translators(
            [&](std::forward_list<ExceptionTranslator> &exception_translators,
                std::forward_list<ExceptionTranslator> &) {
                auto &foreign_internals = *foreign_internals_local_cache();
                if (foreign_internals.exc_frameworks.empty()) {
                    // First foreign framework with an exception translator.
                    // Add our `foreign_exception_translator` wrapper in the
                    // 2nd-last position (last is the default exception
                    // translator).
                    auto leader = exception_translators.begin();
                    auto trailer = exception_translators.before_begin();
                    while (++leader != exception_translators.end()) {
                        ++trailer;
                    }
                    exception_translators.insert_after(trailer, foreign_exception_translator);
                }
                // Add the new framework at the end of the list
                auto it = foreign_internals.exc_frameworks.before_begin();
                while (std::next(it) != foreign_internals.exc_frameworks.end()) {
                    ++it;
                }
                foreign_internals.exc_frameworks.insert_after(it, framework);
            });
    }
}

inline void foreign_cb_remove_foreign_framework(pymb_framework *framework) noexcept {
    // No need for locking; the interpreter is already finalizing
    // at this point (and might be already finalized, so we can't do any
    // Python API calls)
    if (framework->translate_exception) {
        foreign_internals_local_cache()->exc_frameworks.remove(framework);
        // No need to bother removing the foreign_exception_translator if
        // this was the last of the exc_frameworks. In the unlikely event
        // that something needs an exception translated during finalization,
        // it will work fine with an empty exc_frameworks list.
    }
}

// (end of callbacks)

// Advertise our existence, and the above callbacks, to other frameworks
PYBIND11_NOINLINE void foreign_internals::register_with_pymetabind(bool autoimport) {
    pymb_registry *registry = nullptr;
    bool inited_by_us = with_internals([&](internals &) {
        if (self) {
            return false;
        }
        registry = pymb_get_registry();
        if (!registry) {
            throw error_already_set();
        }

        autoimport_for_anyone = autoimport;
        self.reset(new pymb_framework{});
        self->name = "pybind11 " PYBIND11_INTERNALS_ABI_ID;
        self->keep_alive_types
            = ((uint8_t) pymb_keep_alive_callback | (uint8_t) pymb_keep_alive_pyobject
               | (uint8_t) pymb_keep_alive_cpp_shared_ptr_void);
        self->flags = 0;
        self->abi_lang = pymb_abi_lang_cpp;
        self->abi_extra = PYBIND11_PLATFORM_ABI_ID;
        self->from_python = foreign_cb_from_python;
        self->to_python = foreign_cb_to_python;
        self->keep_alive = foreign_cb_keep_alive;
        self->translate_exception = foreign_cb_translate_exception;
        self->remove_local_binding = foreign_cb_remove_local_binding;
        self->free_local_binding = foreign_cb_free_local_binding;
        self->add_foreign_binding = foreign_cb_add_foreign_binding;
        self->remove_foreign_binding = foreign_cb_remove_foreign_binding;
        self->add_foreign_framework = foreign_cb_add_foreign_framework;
        self->remove_foreign_framework = foreign_cb_remove_foreign_framework;
        foreign_internals_local_cache() = this;
        return true;
    });
    if (inited_by_us) {
        // Unlock internals before calling add_framework, so that the callbacks
        // (foreign_cb_add_foreign_binding, etc) can safely re-lock it.
        // Note lock order: pymb_registry lock is 'outside' our internals lock.
        pymb_add_framework(registry, self.get());
    } else if (autoimport) {
        enable_autoimport();
    }
}

inline void foreign_internals::enable_autoimport() {
    if (!self) {
        register_with_pymetabind(/*autoimport=*/true);
        return;
    }

    pymb_registry *registry = nullptr;
    bool enabled_by_us = with_internals([&](internals &) {
        if (autoimport_for_anyone) {
            return false;
        }
        registry = pymb_get_registry();
        if (!registry) {
            throw error_already_set();
        }
        autoimport_for_anyone = true;
        return true;
    });
    if (enabled_by_us) {
        // Note lock order: pymb_registry lock is 'outside' our internals lock.
        pymb_lock_registry(registry);
        // NOLINTNEXTLINE(modernize-use-auto)
        PYMB_LIST_FOREACH(struct pymb_binding *, binding, registry->bindings) {
            if (binding->framework != self.get()) {
                foreign_cb_add_foreign_binding(binding);
            }
        }
        pymb_unlock_registry(registry);
    }
}

inline foreign_internals::~foreign_internals() {
    if (!self) {
        // never set up in the first place
        return;
    }

    // We can only clean up the framework if it has no bindings still active
    bool any_alive = false;
    for (auto &entry : bindings) {
        for (pymb_binding *binding : entry.second) {
            if (binding->framework == self.get()) {
                any_alive = true;
                break;
            }
        }
    }
    if (!any_alive) {
        pymb_remove_framework(self.get());
    } else {
        // Leak framework so the still-existing bindings can be used during
        // teardown of other frameworks
        self.release(); // NOLINT(bugprone-unused-return-value)
    }
    auto &cache = foreign_internals_local_cache();
    if (cache == this) {
        cache = nullptr;
    }
}

// Learn to satisfy from- and to-Python requests for `cpptype` using the
// foreign binding provided by the given `pytype`. If cpptype is nullptr, infer
// the C++ type by looking at the binding, and require that its ABI match ours.
// Throws an exception on failure. Caller must hold the internals lock and have
// already ensured the foreign_internals exist.
PYBIND11_NOINLINE void import_foreign(const std::type_info *cpptype, PyTypeObject *pytype) {
    auto &local_internals = get_local_internals();
    auto &foreign_internals = *local_internals.foreign;
    pymb_binding *binding = pymb_get_binding(reinterpret_cast<PyObject *>(pytype));
    if (!binding) {
        pybind11_fail("pybind11::import_foreign(): type does not define "
                      "a __pymetabind_binding__");
    }
    if (binding->pytype != pytype) {
        pybind11_fail("pybind11::import_foreign(): the binding associated "
                      "with the type you specified is for a different type; "
                      "pass the type object that was created by the other "
                      "framework, not its subclass");
    }
    if (binding->framework == foreign_internals.self.get()) {
        // Can't call get_type_info() because it would lock internals and
        // they're already locked
        auto &internals = get_internals();
        auto it = internals.registered_types_py.find(binding->pytype);
        if (it != internals.registered_types_py.end() && it->second.size() == 1
            && is_local_to_other_module(*it->second.begin())) {
            // Allow importing module-local types from other pybind11 modules,
            // even if they're ABI-compatible with us and thus use the same
            // pymb_framework. The import is not doing much here; the export
            // alone would put the binding in foreign_internals where we can
            // see it.
        } else {
            pybind11_fail("pybind11::import_foreign(): type is not foreign");
        }
    }
    if (!cpptype) {
        if (binding->framework->abi_lang != pymb_abi_lang_cpp) {
            pybind11_fail("pybind11::import_foreign(): type is not "
                          "written in C++, so you must specify a C++ type");
        }
        if (binding->framework->abi_extra != foreign_internals.self->abi_extra) {
            pybind11_fail("pybind11::import_foreign(): type has "
                          "incompatible C++ ABI with this module");
        }
        cpptype = (const std::type_info *) binding->native_type;
    }

    auto result = foreign_internals.manual_imports.emplace(binding, cpptype);
    if (!result.second) {
        const auto *existing = (const std::type_info *) result.first->second;
        if (existing != cpptype && *existing != *cpptype) {
            pybind11_fail("pybind11::import_foreign(): type was "
                          "already imported as a different C++ type");
        }
    }
    if (!local_internals.foreign_import_all) {
        // Mark this binding as having been specifically requested by the
        // current extension module, so that it can bypass the don't-import
        // default
        local_internals.foreign_local_imports.emplace(std::type_index(*cpptype),
                                                      binding->framework);
    }
    import_foreign_binding(binding, cpptype, /*manual=*/true);
}

// Expose hooks for other frameworks to use to work with the given pybind11
// type object. This occurs by default at binding time unless the module was
// created with `py::foreign_interop::import_only()` or a lower level.
// `ti` may be nullptr if exporting a native enum. Caller must hold the
// internals lock and have already ensured the foreign internals exist.
PYBIND11_NOINLINE void
export_to_foreign(const std::type_info *cpptype, PyTypeObject *pytype, type_info *ti) {
    auto &foreign_internals = *get_foreign_internals();
    auto &lst = foreign_internals.bindings[*cpptype];
    for (pymb_binding *existing : lst) {
        if (existing->framework == foreign_internals.self.get() && existing->pytype == pytype) {
            return; // already exported
        }
    }

    auto *binding = new pymb_binding{};
    binding->framework = foreign_internals.self.get();
    binding->pytype = pytype;
    binding->native_type = cpptype;
    binding->source_name = PYBIND11_COMPAT_STRDUP(clean_type_id(cpptype->name()).c_str());
    binding->context = ti;

    ++foreign_internals.bindings_update_count;
    lst.push_back(binding);

#ifdef Py_GIL_DISABLED
    // Call pymb_add_binding() with unlocked internals in order to maintain
    // consistent lock order: the pymb_registry lock is locked outside our
    // internals lock in enable_autoimport(), so it must not be locked inside
    // our internals lock here. pymb_add_binding() is noexcept so we don't
    // need a scope guard.
    auto &internals = get_internals();
    internals.mutex.unlock();
#endif
    pymb_add_binding(binding, /* tp_finalize_will_remove */ 0);
#ifdef Py_GIL_DISABLED
    internals.mutex.lock();
#endif
}

// Invoke `attempt(closure, binding)` for each foreign binding `binding`
// that claims `type` and was not supplied by us, until one of them returns
// non-null. Return that first non-null value, or null if all attempts failed.
// Caller attests they have already checked that the foreign internals exist.
PYBIND11_NOINLINE void *try_foreign_bindings(const std::type_info *type,
                                             void *(*attempt)(const void *closure,
                                                              pymb_binding *binding),
                                             const void *closure) {
    auto &internals = get_internals();
    auto &local_internals = get_local_internals();
    auto &foreign_internals = *local_internals.foreign;
    uint32_t update_count = foreign_internals.bindings_update_count;

    do {
        PYBIND11_LOCK_INTERNALS(internals);
        (void) internals; // suppress unused warning on non-ft builds
        auto it = foreign_internals.bindings.find(*type);
        if (it == foreign_internals.bindings.end()) {
            return nullptr;
        }
        for (pymb_binding *binding : it->second) {
            if (binding->framework == foreign_internals.self.get()
                && (!binding->context
                    || !is_local_to_other_module((type_info *) binding->context))) {
                // Don't try to use our own types, unless they're module-local
                // to some other module and this is the only way we'd see them.
                // (The module-local escape hatch is only relevant for
                // to-Python conversions; from-Python won't try foreign if it
                // sees the capsule for other-module-local.)
                continue;
            }

            if (!local_internals.foreign_import_all) {
                // If this extension module disabled automatic import of foreign
                // bindings, then don't consider any unless explicitly imported.
                auto range
                    = local_internals.foreign_local_imports.equal_range(std::type_index(*type));
                bool found = false;
                for (auto it2 = range.first; it2 != range.second; ++it2) {
                    if (it2->second == binding->framework) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    continue;
                }
            }

            {
#ifdef Py_GIL_DISABLED
                // attempt() might execute Python code; drop the internals lock
                // to avoid a deadlock
                auto guard = lock.temporarily_drop();
#endif
                void *result = attempt(closure, binding);
                if (result) {
                    return result;
                }
            }
            // Make sure our iterator wasn't invalidated by something that
            // was done within attempt(), or concurrently during attempt()
            // while we didn't hold the internals lock
            if (foreign_internals.bindings_update_count != update_count) {
                // Concurrent update occurred; stop iterating
                break;
            }
        }
        if (foreign_internals.bindings_update_count != update_count) {
            // We broke out early due to a concurrent update. Retry from the top.
            update_count = foreign_internals.bindings_update_count;
            continue;
        }
        return nullptr;
    } while (true);
}

// Friendlier version which calls `attempt(binding)` for each binding,
// with captures carried in the lambda and adding checking for whether
// foreign types are enabled.
template <class Fn>
inline void *try_foreign_bindings(const std::type_info *type, const Fn &attempt) {
    auto &local_internals = get_local_internals();
    if (!local_internals.foreign) {
        return nullptr;
    }
    return try_foreign_bindings(
        type,
        +[](const void *closure, pymb_binding *binding) {
            return (*static_cast<const Fn *>(closure))(binding);
        },
        &attempt);
}

inline foreign_internals &require_foreign_internals() {
    auto *result = get_foreign_internals();
    if (!result) {
        pybind11_fail("foreign import/export are not supported in an extension "
                      "module that used the py::foreign_interop::disabled() "
                      "module option");
    }
    return *result;
}

PYBIND11_NAMESPACE_END(detail)

template <class T = void>
inline void import_foreign(handle pytype) {
    if (!PyType_Check(pytype.ptr())) {
        pybind11_fail("pybind11::import_foreign(): expected a type object");
    }
    const std::type_info *cpptype = std::is_void<T>::value ? nullptr : &typeid(T);
    auto &foreign_internals = detail::require_foreign_internals();
    detail::with_internals([&](detail::internals &) {
        foreign_internals.import_foreign(cpptype, (PyTypeObject *) pytype.ptr());
    });
}

inline void export_to_foreign(handle ty) {
    if (!PyType_Check(ty.ptr())) {
        pybind11_fail("pybind11::export_foreign(): expected a type object");
    }
    auto &foreign_internals = detail::require_foreign_internals();
    detail::type_info *ti = detail::get_type_info((PyTypeObject *) ty.ptr());
    if (ti) {
        detail::with_internals([&](detail::internals &) {
            foreign_internals.export_to_foreign(ti->cpptype, ti->type, ti);
        });
        return;
    }
    // Not a class_; maybe it's a native_enum?
    try {
        auto cap
            = reinterpret_borrow<capsule>(ty.attr(detail::native_enum_record::attribute_name()));
        auto *info = cap.get_pointer<detail::native_enum_record>();
        bool ours = detail::with_internals([&](detail::internals &internals) {
            auto it = internals.native_enum_type_map.find(*info->cpptype);
            if (it != internals.native_enum_type_map.end() && it->second == ty.ptr()) {
                foreign_internals.export_to_foreign(
                    info->cpptype, (PyTypeObject *) ty.ptr(), nullptr);
                return true;
            }
            return false;
        });
        if (ours) {
            return;
        }
    } catch (error_already_set &) { // NOLINT(bugprone-empty-catch)
        // Could be an older native enum without __pybind11_enum__ capsule
    }
    pybind11_fail("pybind11::export_to_foreign: not a "
                  "pybind11 class or enum bound in this domain");
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
