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

// pybind11 exception translator that tries all known foreign ones
PYBIND11_NOINLINE void foreign_exception_translator(std::exception_ptr p) {
    auto &interop_internals = get_interop_internals();
    for (pymb_framework *fw : interop_internals.exc_frameworks) {
        if (fw->translate_exception(&p) != 0) {
            return;
        }
    }
    std::rethrow_exception(p);
}

// When learning about a new foreign type, should we automatically use it?
inline bool should_autoimport_foreign(interop_internals &interop_internals,
                                      pymb_binding *binding) {
    return interop_internals.import_all && binding->framework->abi_lang == pymb_abi_lang_cpp
           && binding->framework->abi_extra == interop_internals.self->abi_extra;
}

// Determine whether a pybind11 type is module-local from a different module
inline bool is_local_to_other_module(type_info *ti) {
    return ti->module_local_load != nullptr
           && ti->module_local_load != &type_caster_generic::local_load;
}

// Add the given `binding` to our type maps so that we can use it to satisfy
// from- and to-Python requests for the given C++ type
inline void import_foreign_binding(pymb_binding *binding, const std::type_info *cpptype) noexcept {
    // Caller must hold the internals lock
    auto &interop_internals = get_interop_internals();
    interop_internals.imported_any = true;
    auto &lst = interop_internals.bindings[*cpptype];
    for (pymb_binding *existing : lst) {
        if (existing == binding) {
            return; // already imported
        }
    }
    ++interop_internals.bindings_update_count;
    lst.append(binding);
}

// Callback functions for other frameworks to operate on our objects
// or tell us about theirs

inline void *interop_cb_from_python(pymb_binding *binding,
                                    PyObject *pyobj,
                                    uint8_t convert,
                                    void (*keep_referenced)(void *ctx, PyObject *obj),
                                    void *keep_referenced_ctx) noexcept {
    if (binding->context == nullptr) {
        // This is a native enum type. We can only return a pointer to the C++
        // enum if we're able to allocate a temporary.
        handle pytype((PyObject *) binding->pytype);
        if (!keep_referenced || !isinstance(pyobj, pytype)) {
            return nullptr;
        }
        try {
            auto cap
                = reinterpret_borrow<capsule>(pytype.attr(native_enum_record::attribute_name()));
            auto *info = cap.get_pointer<native_enum_record>();
            auto value = handle(pyobj).attr("value");
            uint64_t ival = 0;
            if (info->is_signed && handle(value) < int_(0)) {
                ival = (uint64_t) cast<int64_t>(value);
            } else {
                ival = cast<uint64_t>(value);
            }
            bytes holder{reinterpret_cast<const char *>(&ival)
                             + PYBIND11_BIG_ENDIAN * size_t(8 - info->size_bytes),
                         info->size_bytes};
            keep_referenced(keep_referenced_ctx, holder.ptr());
            return PyBytes_AsString(holder.ptr());
        } catch (error_already_set &exc) {
            exc.discard_as_unraisable("Error converting native enum from Python");
            return nullptr;
        }
    }

#if defined(PYBIND11_HAS_OPTIONAL)
    using maybe_life_support = std::optional<loader_life_support>;
#else
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
// inhibits registration of the new instance so that interop_cb_keep_alive()
// can fix up the holder before other threads start using the new instance.
inline void init_instance_unregistered(instance *inst, const void *holder) {
    assert(holder == nullptr && !inst->owned);
    (void) holder; // avoid unused warning if compiled without asserts
    value_and_holder v_h = *values_and_holders(inst).begin();

    // If using smart_holder, force creation of a shared_ptr that has a
    // guarded_delete deleter, so that we can modify it in
    // interop_cb_keep_alive(). We can't create it there because it needs to be
    // created in the same DSO as it's accessed in; init_instance is in that
    // DSO, but this function might not be.
    if (v_h.type->holder_enum_v == holder_enum_t::smart_holder) {
        inst->owned = true;
    }

    // Pretend it's already registered so that init_instance doesn't try again
    v_h.set_instance_registered(true);

    // Undo our shenanigans even if init_instance raises an exception
    struct guard {
        value_and_holder &v_h;
        ~guard() noexcept {
            v_h.set_instance_registered(false);
            if (v_h.type->holder_enum_v == holder_enum_t::smart_holder) {
                v_h.inst->owned = false;
                auto &h = v_h.holder<smart_holder>();
                h.vptr_is_using_std_default_delete = true;
                h.reset_vptr_deleter_armed_flag(v_h.type->get_memory_guarded_delete,
                                                /* armed_flag */ false);
            }
        }
    } guard{v_h};
    v_h.type->init_instance(inst, nullptr);
}

inline PyObject *interop_cb_to_python(pymb_binding *binding,
                                      void *cobj,
                                      enum pymb_rv_policy rvp_,
                                      pymb_to_python_feedback *feedback) noexcept {
    feedback->relocate = 0; // we don't support relocation
    feedback->is_new = 0;   // unless overridden below

    if (cobj == nullptr) {
        return none().release().ptr();
    }

    if (!binding->context) {
        // Native enum type
        try {
            handle pytype((PyObject *) binding->pytype);
            auto cap
                = reinterpret_borrow<capsule>(pytype.attr(native_enum_record::attribute_name()));
            auto *info = cap.get_pointer<native_enum_record>();
            uint64_t key = 0;
            switch (info->size_bytes) {
                case 1:
                    key = *(uint8_t *) cobj;
                    break;
                case 2:
                    key = *(uint16_t *) cobj;
                    break;
                case 4:
                    key = *(uint32_t *) cobj;
                    break;
                case 8:
                    key = *(uint64_t *) cobj;
                    break;
                default:
                    return nullptr;
            }
            if (rvp_ == pymb_rv_policy_take_ownership) {
                ::operator delete(cobj);
            }
            if (info->is_signed) {
                auto ikey = (int64_t) key;
                if (info->size_bytes < 8) {
                    // sign extend
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

    const auto *ti = static_cast<const type_info *>(binding->context);
    return_value_policy rvp = return_value_policy::automatic;
    bool inhibit_registration = false;

    switch (rvp_) {
        case pymb_rv_policy_take_ownership:
        case pymb_rv_policy_copy:
        case pymb_rv_policy_move:
        case pymb_rv_policy_reference:
            // These have the same values and semantics as our own policies
            rvp = (return_value_policy) rvp_;
            break;
        case pymb_rv_policy_share_ownership:
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
        with_internals([&](internals &) {
            auto &interop_internals = get_interop_internals();
            auto it = interop_internals.copy_move_ctors.find(*ti->cpptype);
            if (it != interop_internals.copy_move_ctors.end()) {
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
        feedback->is_new = uint8_t(srcs.is_new);
        return ret.ptr();
    } catch (...) {
        translate_exception(std::current_exception());
        return nullptr;
    }
}

inline int interop_cb_keep_alive(PyObject *nurse, void *payload, void (*cb)(void *)) noexcept {
    try {
        do { // single-iteration loop to reduce nesting level
            if (!is_uniquely_referenced(nurse)) {
                break;
            }
            // See if we can install this as a shared_ptr deleter rather than
            // a keep_alive, since the very first keep_alive for a new object
            // might be to let it carry shared_ptr ownership. This helps
            // a shared_ptr<T> returned from a foreign binding be acceptable
            // as a shared_ptr<T> argument to a pybind11-bound function.
            values_and_holders vhs{nurse};
            if (vhs.size() != 1) {
                break;
            }
            value_and_holder v_h = *vhs.begin();
            if (v_h.instance_registered()) {
                break;
            }
            auto cb_to_use = cb ? cb : (decltype(cb)) Py_DecRef;
            bool success = false;
            if (v_h.type->holder_enum_v == holder_enum_t::std_shared_ptr
                && !v_h.holder_constructed()) {
                // Create a shared_ptr whose destruction will perform the action
                std::shared_ptr<void> owner(payload, cb_to_use);
                // Use the aliasing constructor to make its get() return the right thing
                // NB: this constructor accepts an rvalue reference only in C++20
                new (std::addressof(v_h.holder<std::shared_ptr<void>>()))
                    // NOLINTNEXTLINE(performance-move-const-arg)
                    std::shared_ptr<void>(std::move(owner), v_h.value_ptr());
                v_h.set_holder_constructed();
                success = true;
            } else if (v_h.type->holder_enum_v == holder_enum_t::smart_holder
                       && v_h.holder_constructed() && !v_h.inst->owned) {
                auto &h = v_h.holder<smart_holder>();
                auto *gd = v_h.type->get_memory_guarded_delete(h.vptr);
                if (gd && !gd->armed_flag) {
                    gd->del_fun = [=](void *) { cb_to_use(payload); };
                    gd->use_del_fun = true;
                    gd->armed_flag = true;
                    success = true;
                }
            }
            register_instance(v_h.inst, v_h.value_ptr(), v_h.type);
            v_h.set_instance_registered(true);
            if (success) {
                return 1;
            }
        } while (false);

        if (!cb) {
            keep_alive_impl(nurse, static_cast<PyObject *>(payload));
        } else {
            capsule patient{payload, cb};
            keep_alive_impl(nurse, patient);
        }
        return 1;
    } catch (...) {
        translate_exception(std::current_exception());
        PyErr_WriteUnraisable(nurse);
        return 0;
    }
}

inline int interop_cb_translate_exception(void *eptr) noexcept {
    return with_exception_translators(
        [&](std::forward_list<ExceptionTranslator> &exception_translators,
            std::forward_list<ExceptionTranslator> & /*local_exception_translators*/) {
            // Ignore local exception translators. We're being called to translate
            // an exception that was raised from a different framework, thus a
            // different extension module, so nothing local to us will apply.
            // Try global translators, except the last one or two.
            std::exception_ptr &e = *(std::exception_ptr *) eptr;
            auto it = exception_translators.begin();
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
            if (!get_interop_internals().exc_frameworks.empty()) {
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

inline void interop_cb_remove_local_binding(pymb_binding *binding) noexcept {
    with_internals([&](internals &) {
        auto &interop_internals = get_interop_internals();
        const auto *cpptype = (const std::type_info *) binding->native_type;
        auto it = interop_internals.bindings.find(*cpptype);
        if (it != interop_internals.bindings.end() && it->second.erase(binding)) {
            ++interop_internals.bindings_update_count;
            if (it->second.empty()) {
                interop_internals.bindings.erase(it);
            }
        }
    });
}

inline void interop_cb_free_local_binding(pymb_binding *binding) noexcept {
    free(const_cast<char *>(binding->source_name));
    delete binding;
}

inline void interop_cb_add_foreign_binding(pymb_binding *binding) noexcept {
    with_internals([&](internals &) {
        auto &interop_internals = get_interop_internals();
        if (should_autoimport_foreign(interop_internals, binding)) {
            import_foreign_binding(binding, (const std::type_info *) binding->native_type);
        }
    });
}

inline void interop_cb_remove_foreign_binding(pymb_binding *binding) noexcept {
    with_internals([&](internals &) {
        auto &interop_internals = get_interop_internals();
        auto remove_from_type = [&](const std::type_info *type) {
            auto it = interop_internals.bindings.find(*type);
            if (it != interop_internals.bindings.end() && it->second.erase(binding)) {
                ++interop_internals.bindings_update_count;
                if (it->second.empty()) {
                    interop_internals.bindings.erase(it);
                }
            }
        };
        bool should_remove_auto = should_autoimport_foreign(interop_internals, binding);
        auto it = interop_internals.manual_imports.find(binding);
        if (it != interop_internals.manual_imports.end()) {
            remove_from_type(it->second);
            should_remove_auto &= (it->second != binding->native_type);
            interop_internals.manual_imports.erase(it);
        }
        if (should_remove_auto) {
            remove_from_type((const std::type_info *) binding->native_type);
        }
    });
}

inline void interop_cb_add_foreign_framework(pymb_framework *framework) noexcept {
    if (framework->translate_exception) {
        with_exception_translators(
            [&](std::forward_list<ExceptionTranslator> &exception_translators,
                std::forward_list<ExceptionTranslator> &) {
                auto &interop_internals = get_interop_internals();
                if (interop_internals.exc_frameworks.empty()) {
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
                auto it = interop_internals.exc_frameworks.before_begin();
                while (std::next(it) != interop_internals.exc_frameworks.end()) {
                    ++it;
                }
                interop_internals.exc_frameworks.insert_after(it, framework);
            });
    }
}

inline void interop_cb_remove_foreign_framework(pymb_framework *framework) noexcept {
    // No need for locking; the interpreter is already finalizing
    // at this point (and might be already finalized, so we can't do any
    // Python API calls)
    if (framework->translate_exception) {
        get_interop_internals().exc_frameworks.remove(framework);
        // No need to bother removing the foreign_exception_translator if
        // this was the last of the exc_frameworks. In the unlikely event
        // that something needs an exception translated during finalization,
        // it will work fine with an empty exc_frameworks list.
    }
}

// (end of callbacks)

// Advertise our existence, and the above callbacks, to other frameworks
PYBIND11_NOINLINE bool interop_internals::initialize() {
    pymb_registry *registry = nullptr;
    bool inited_by_us = with_internals([&](internals &) {
        if (self) {
            return false;
        }
        registry = pymb_get_registry();
        if (!registry) {
            throw error_already_set();
        }

        self.reset(new pymb_framework{});
        self->name = "pybind11 " PYBIND11_ABI_TAG;
        self->flags = 0;
        self->abi_lang = pymb_abi_lang_cpp;
        self->abi_extra = PYBIND11_PLATFORM_ABI_ID;
        self->from_python = interop_cb_from_python;
        self->to_python = interop_cb_to_python;
        self->keep_alive = interop_cb_keep_alive;
        self->translate_exception = interop_cb_translate_exception;
        self->remove_local_binding = interop_cb_remove_local_binding;
        self->free_local_binding = interop_cb_free_local_binding;
        self->add_foreign_binding = interop_cb_add_foreign_binding;
        self->remove_foreign_binding = interop_cb_remove_foreign_binding;
        self->add_foreign_framework = interop_cb_add_foreign_framework;
        self->remove_foreign_framework = interop_cb_remove_foreign_framework;
        return true;
    });
    if (inited_by_us) {
        // Unlock internals before calling add_framework, so that the callbacks
        // (interop_cb_add_foreign_binding, etc) can safely re-lock it.
        pymb_add_framework(registry, self.get());
    }
    return inited_by_us;
}

inline interop_internals::~interop_internals() {
    if (self && bindings.empty()) {
        pymb_remove_framework(self.get());
    }
}

// Learn to satisfy from- and to-Python requests for `cpptype` using the
// foreign binding provided by the given `pytype`. If cpptype is nullptr, infer
// the C++ type by looking at the binding, and require that its ABI match ours.
// Throws an exception on failure. Caller must hold the internals lock and have
// already called interop_internals.initialize_if_needed().
PYBIND11_NOINLINE void import_for_interop(handle pytype, const std::type_info *cpptype) {
    auto &interop_internals = get_interop_internals();
    pymb_binding *binding = pymb_get_binding(pytype.ptr());
    if (!binding) {
        pybind11_fail("pybind11::import_for_interop(): type does not define "
                      "a __pymetabind_binding__");
    }
    if (binding->pytype != (PyTypeObject *) pytype.ptr()) {
        pybind11_fail("pybind11::import_for_interop(): the binding associated "
                      "with the type you specified is for a different type; "
                      "pass the type object that was created by the other "
                      "framework, not its subclass");
    }
    if (binding->framework == interop_internals.self.get()) {
        // Can't call get_type_info() because it would lock internals and
        // they're already locked
        auto &internals = get_internals();
        auto it = internals.registered_types_py.find(binding->pytype);
        if (it != internals.registered_types_py.end() && it->second.size() == 1
            && is_local_to_other_module(*it->second.begin())) {
            // Allow importing module-local types from other pybind11 modules,
            // even if they're ABI-compatible with us and thus use the same
            // pymb_framework. The import is not doing much here; the export
            // alone would put the binding in interop_internals where we can
            // see it.
        } else {
            pybind11_fail("pybind11::import_for_interop(): type is not foreign");
        }
    }
    if (!cpptype) {
        if (binding->framework->abi_lang != pymb_abi_lang_cpp) {
            pybind11_fail("pybind11::import_for_interop(): type is not "
                          "written in C++, so you must specify a C++ type");
        }
        if (binding->framework->abi_extra != interop_internals.self->abi_extra) {
            pybind11_fail("pybind11::import_for_interop(): type has "
                          "incompatible C++ ABI with this module");
        }
        cpptype = (const std::type_info *) binding->native_type;
    }

    auto result = interop_internals.manual_imports.emplace(binding, cpptype);
    if (!result.second) {
        const auto *existing = (const std::type_info *) result.first->second;
        if (existing != cpptype && *existing != *cpptype) {
            pybind11_fail("pybind11::import_for_interop(): type was "
                          "already imported as a different C++ type");
        }
    }
    import_foreign_binding(binding, cpptype);
}

// Call `import_foreign_binding()` for every ABI-compatible type provided by
// other C++ binding frameworks used by extension modules loaded in this
// interpreter, both those that exist now and those bound in the future.
PYBIND11_NOINLINE void interop_enable_import_all() {
    auto &interop_internals = get_interop_internals();
    bool proceed = with_internals([&](internals &) {
        if (interop_internals.import_all) {
            return false;
        }
        interop_internals.import_all = true;
        return true;
    });
    if (!proceed) {
        return;
    }
    if (interop_internals.initialize_if_needed()) {
        // pymb_add_framework tells us about every existing type when we
        // register, so if we register with import enabled, we're done
        return;
    }
    // If we enable import after registering, we have to iterate over the
    // list of types ourselves. Do this without the internals lock held so
    // we can reuse the pymb callback functions. interop_internals registry +
    // self never change once they're non-null, so we can access them
    // without locking here.
    struct pymb_registry *registry = interop_internals.self->registry;
    pymb_lock_registry(registry);
    // NOLINTNEXTLINE(modernize-use-auto)
    PYMB_LIST_FOREACH(struct pymb_binding *, binding, registry->bindings) {
        if (binding->framework != interop_internals.self.get()) {
            interop_cb_add_foreign_binding(binding);
        }
    }
    pymb_unlock_registry(registry);
}

// Expose hooks for other frameworks to use to work with the given pybind11
// type object. `ti` may be nullptr if exporting a native enum.
// Caller must hold the internals lock and have already called
// interop_internals.initialize_if_needed().
PYBIND11_NOINLINE void
export_for_interop(const std::type_info *cpptype, PyTypeObject *pytype, type_info *ti) {
    auto &interop_internals = get_interop_internals();
    auto &lst = interop_internals.bindings[*cpptype];
    for (pymb_binding *existing : lst) {
        if (existing->framework == interop_internals.self.get() && existing->pytype == pytype) {
            return; // already imported
        }
    }

    auto *binding = new pymb_binding{};
    binding->framework = interop_internals.self.get();
    binding->pytype = pytype;
    binding->native_type = cpptype;
    binding->source_name = PYBIND11_COMPAT_STRDUP(clean_type_id(cpptype->name()).c_str());
    binding->context = ti;

    ++interop_internals.bindings_update_count;
    lst.append(binding);
    pymb_add_binding(binding, /* tp_finalize_will_remove */ 0);
}

// Call `export_type_to_foreign()` for each type that currently exists in our
// internals structure and each type created in the future.
PYBIND11_NOINLINE void interop_enable_export_all() {
    auto &interop_internals = get_interop_internals();
    bool proceed = with_internals([&](internals &) {
        if (interop_internals.export_all) {
            return false;
        }
        interop_internals.export_all = true;
        interop_internals.export_for_interop = &detail::export_for_interop;
        return true;
    });
    if (!proceed) {
        return;
    }
    interop_internals.initialize_if_needed();
    with_internals([&](internals &internals) {
        for (const auto &entry : internals.registered_types_cpp) {
            auto *ti = entry.second;
            detail::export_for_interop(ti->cpptype, ti->type, ti);
        }
        for (const auto &entry : internals.native_enum_type_map) {
            try {
                auto cap = reinterpret_borrow<capsule>(
                    handle(entry.second).attr(native_enum_record::attribute_name()));
                auto *info = cap.get_pointer<native_enum_record>();
                detail::export_for_interop(info->cpptype, (PyTypeObject *) entry.second, nullptr);
            } catch (error_already_set &) { // NOLINT(bugprone-empty-catch)
                // Ignore native enums without a __pybind11_enum__ capsule;
                // they might be from an older version of pybind11
            }
        }
    });
}

// Invoke `attempt(closure, binding)` for each foreign binding `binding`
// that claims `type` and was not supplied by us, until one of them returns
// non-null. Return that first non-null value, or null if all attempts failed.
PYBIND11_NOINLINE void *try_foreign_bindings(const std::type_info *type,
                                             void *(*attempt)(void *closure,
                                                              pymb_binding *binding),
                                             void *closure) {
    auto &internals = get_internals();
    auto &interop_internals = get_interop_internals();
    uint32_t update_count = interop_internals.bindings_update_count;

    do {
        PYBIND11_LOCK_INTERNALS(internals);
        (void) internals; // suppress unused warning on non-ft builds
        auto it = interop_internals.bindings.find(*type);
        if (it == interop_internals.bindings.end()) {
            return nullptr;
        }
        for (pymb_binding *binding : it->second) {
            if (binding->framework == interop_internals.self.get()
                && (!binding->context
                    || !is_local_to_other_module((type_info *) binding->context))) {
                // Don't try to use our own types, unless they're module-local
                // to some other module and this is the only way we'd see them.
                // (The module-local escape hatch is only relevant for
                // to-Python conversions; from-Python won't try foreign if it
                // sees the capsule for other-module-local.)
                continue;
            }

#ifdef Py_GIL_DISABLED
            // attempt() might execute Python code; drop the internals lock
            // to avoid a deadlock
            lock.unlock();
#endif
            void *result = attempt(closure, binding);
            if (result) {
                return result;
            }
#ifdef Py_GIL_DISABLED
            lock.lock();
#endif
            // Make sure our iterator wasn't invalidated by something that
            // was done within attempt(), or concurrently during attempt()
            // while we didn't hold the internals lock
            if (interop_internals.bindings_update_count != update_count) {
                // Concurrent update occurred; stop iterating
                break;
            }
        }
        if (interop_internals.bindings_update_count != update_count) {
            // We broke out early due to a concurrent update. Retry from the top.
            update_count = interop_internals.bindings_update_count;
            continue;
        }
        return nullptr;
    } while (true);
}

PYBIND11_NAMESPACE_END(detail)

inline void interoperate_by_default(bool export_all = true, bool import_all = true) {
    auto &interop_internals = detail::get_interop_internals();
    if (import_all && !interop_internals.import_all) {
        detail::interop_enable_import_all();
    }
    if (export_all && !interop_internals.export_all) {
        detail::interop_enable_export_all();
    }
}

template <class T = void>
inline void import_for_interop(handle pytype) {
    if (!PyType_Check(pytype.ptr())) {
        pybind11_fail("pybind11::import_for_interop(): expected a type object");
    }
    const std::type_info *cpptype = std::is_void<T>::value ? nullptr : &typeid(T);
    auto &interop_internals = detail::get_interop_internals();
    interop_internals.initialize_if_needed();
    detail::with_internals(
        [&](detail::internals &) { detail::import_for_interop(pytype, cpptype); });
}

inline void export_for_interop(handle ty) {
    if (!PyType_Check(ty.ptr())) {
        pybind11_fail("pybind11::export_for_interop(): expected a type object");
    }
    auto &interop_internals = detail::get_interop_internals();
    interop_internals.initialize_if_needed();
    detail::type_info *ti = detail::get_type_info((PyTypeObject *) ty.ptr());
    if (ti) {
        detail::with_internals(
            [&](detail::internals &) { detail::export_for_interop(ti->cpptype, ti->type, ti); });
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
                detail::export_for_interop(info->cpptype, (PyTypeObject *) ty.ptr(), nullptr);
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
    pybind11_fail("pybind11::export_for_interop: not a "
                  "pybind11 class or enum bound in this domain");
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
