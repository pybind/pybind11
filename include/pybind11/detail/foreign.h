/*
    pybind11/detail/foreign.h: Interoperability with other binding frameworks

    Copyright (c) 2025 Hudson River Trading <opensource@hudson-trading.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"
#include "internals.h"
#include "type_caster_base.h"
#include "pymetabind.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

// pybind11 exception translator that tries all known foreign ones
PYBIND11_NOINLINE void foreign_exception_translator(std::exception_ptr p) {
    auto& foreign_internals = get_foreign_internals();
    for (pymb_framework *fw : foreign_internals.exc_frameworks) {
        try {
            fw->translate_exception(&p);
        } catch (...) {
            p = std::current_exception();
        }
    }
    std::rethrow_exception(p);
}

// When learning about a new foreign type, should we automatically use it?
inline bool should_autoimport_foreign(foreign_internals &foreign_internals,
                                      pymb_binding *binding) {
    return foreign_internals.import_all &&
           binding->framework->abi_lang == pymb_abi_lang_cpp &&
           binding->framework->abi_extra == foreign_internals.self->abi_extra;
}

// Add the given `binding` to our type maps so that we can use it to satisfy
// from- and to-Python requests for the given C++ type
inline void import_foreign_binding(pymb_binding *binding,
                                   const std::type_info *cpptype) noexcept {
    // Caller must hold the internals lock
    auto &foreign_internals = get_foreign_internals();
    foreign_internals.imported_any = true;
    foreign_internals.bindings.emplace(*cpptype, binding);
}

// Callback functions for other frameworks to operate on our objects
// or tell us about theirs

inline void *foreign_cb_from_python(pymb_binding *binding,
                                    PyObject *pyobj,
                                    uint8_t convert,
                                    void (*keep_referenced)(void *ctx,
                                                            PyObject *obj),
                                    void *keep_referenced_ctx) noexcept {
#if defined(PYBIND11_HAS_OPTIONAL)
    using maybe_life_support = std::optional<loader_life_support>;
#else
    struct maybe_life_support {
        union {
            loader_life_support supp;
        };
        bool engaged = false;

        maybe_life_support() {}
        maybe_life_support(maybe_life_support&) = delete;
        loader_life_support* operator->() { return &supp; }
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
    type_caster_generic caster{static_cast<const type_info*>(binding->context)};
    void* ret = nullptr;
    try {
        if (caster.load(pyobj, convert)) {
            ret = caster.value;
        }
    } catch (...) {
        translate_exception(std::current_exception());
        PyErr_WriteUnraisable(pyobj);
    }
    if (keep_referenced) {
        for (PyObject *item : holder->list_patients()) {
            keep_referenced(keep_referenced_ctx, item);
        }
    }
    return ret;
}

inline PyObject *foreign_cb_to_python(pymb_binding *binding,
                                      void *cobj,
                                      enum pymb_rv_policy rvp_,
                                      PyObject *parent) noexcept {
    auto* ti = static_cast<const type_info*>(binding->context);
    if (cobj == nullptr) {
        return none().release().ptr();
    }
    auto rvp = static_cast<return_value_policy>(rvp_);
    if (rvp > return_value_policy::reference_internal) {
        // Treat out-of-range rvp as "return existing instance but don't
        // make a new one", for compatibility with pymb_rv_policy_none
        return find_registered_python_instance(cobj, ti).ptr();
    }

    copy_or_move_ctor copy_ctor = nullptr, move_ctor = nullptr;
    if (rvp == return_value_policy::copy || rvp == return_value_policy::move) {
        with_internals([&](internals&) {
            auto& foreign_internals = get_foreign_internals();
            auto it = foreign_internals.copy_move_ctors.find(*ti->cpptype);
            if (it != foreign_internals.copy_move_ctors.end()) {
                std::tie(copy_ctor, move_ctor) = it->second;
            }
        });
    }

    try {
        return type_caster_generic::cast(cobj, rvp, parent, ti,
                                         copy_ctor, move_ctor).ptr();
    } catch (...) {
        translate_exception(std::current_exception());
        return nullptr;
    }
}

inline int foreign_cb_keep_alive(PyObject *nurse,
                                 void *payload,
                                 void (*cb)(void*)) noexcept {
    try {
        if (!cb) {
            keep_alive_impl(nurse, static_cast<PyObject*>(payload));
        } else {
            capsule patient{payload, cb};
            keep_alive_impl(nurse, patient);
        }
        return 0;
    } catch (...) {
        translate_exception(std::current_exception());
        return -1;
    }
}

inline void foreign_cb_translate_exception(const void *eptr) {
    with_exception_translators(
        [&](std::forward_list<ExceptionTranslator> &exception_translators,
            std::forward_list<ExceptionTranslator> &local_exception_translators) {
            // Try local translators. These don't have any special entries
            // we need to skip.
            std::exception_ptr e = *(const std::exception_ptr *) eptr;
            for (auto &translator : local_exception_translators) {
                try {
                    translator(e);
                    return;
                } catch (...) {
                    e = std::current_exception();
                }
            }

            // Try global translators, except the last one or two.
            e = *(const std::exception_ptr *) eptr;
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
            if (!get_foreign_internals().exc_frameworks.empty())
                ++leader;

            for (; leader != exception_translators.end(); ++it, ++leader) {
                try {
                    (*it)(e);
                    return;
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
                return;
            } catch (const builtin_exception &err) {
                // Could not use template since it's an abstract class.
                if (const auto *nep = dynamic_cast<const std::nested_exception *>(
                            std::addressof(err))) {
                    handle_nested_exception(*nep, e);
                }
                err.set_error();
                return;
            }
            // Anything not caught by the above bubbles out.
        });
}

inline void foreign_cb_add_foreign_binding(pymb_binding *binding) noexcept {
    with_internals([&](internals&) {
        auto& foreign_internals = get_foreign_internals();
        if (should_autoimport_foreign(foreign_internals, binding)) {
            import_foreign_binding(
                    binding, (const std::type_info *) binding->native_type);
        }
    });
}

inline void foreign_cb_remove_foreign_binding(pymb_binding *binding) noexcept {
    with_internals([&](internals&) {
        auto& foreign_internals = get_foreign_internals();
        auto remove_from_type = [&](const std::type_info *type) {
            auto range = foreign_internals.bindings.equal_range(*type);
            for (auto it = range.first; it != range.second; ++it) {
                if (it->second == binding) {
                    foreign_internals.bindings.erase(it);
                    break;
                }
            }
        };
        bool should_remove_auto =
            should_autoimport_foreign(foreign_internals, binding);
        auto it = foreign_internals.manual_imports.find(binding);
        if (it != foreign_internals.manual_imports.end()) {
            remove_from_type(it->second);
            should_remove_auto &= (it->second != binding->native_type);
            foreign_internals.manual_imports.erase(it);
        }
        if (should_remove_auto)
            remove_from_type((const std::type_info *) binding->native_type);
    });
}

inline void foreign_cb_add_foreign_framework(pymb_framework *framework)
        noexcept {
    if (framework->translate_exception) {
        with_exception_translators(
            [&](std::forward_list<ExceptionTranslator> &exception_translators,
                std::forward_list<ExceptionTranslator> &) {
                auto& foreign_internals = get_foreign_internals();
                if (foreign_internals.exc_frameworks.empty()) {
                    // First foreign framework with an exception translator.
                    // Add our `foreign_exception_translator` wrapper in the
                    // 2nd-last position (last is the default exception
                    // translator).
                    auto leader = exception_translators.begin();
                    auto trailer = exception_translators.before_begin();
                    while (++leader != exception_translators.end())
                        ++trailer;
                    exception_translators.insert_after(
                            trailer, foreign_exception_translator);
                }
                // Add the new framework at the end of the list
                auto leader = foreign_internals.exc_frameworks.begin();
                auto trailer = leader;
                while (++leader != foreign_internals.exc_frameworks.end())
                    ++trailer;
                foreign_internals.exc_frameworks.insert_after(
                        trailer, framework);
            });
    }
}

// (end of callbacks)

// Advertise our existence, and the above callbacks, to other frameworks
PYBIND11_NOINLINE bool foreign_internals::initialize() {
    bool inited_by_us = with_internals([&](internals&) {
        if (registry)
            return false;
        registry = pymb_get_registry();
        if (!registry)
            throw error_already_set();

        self = std::make_unique<pymb_framework>();
        self->name = "pybind11 " PYBIND11_ABI_TAG;
        // TODO: pybind11 does leak some bindings; there should be a way to
        // indicate that (so that eg nanobind can disable its leak detection)
        // without promising to leak all bindings
        self->bindings_usable_forever = 0;
        self->abi_lang = pymb_abi_lang_cpp;
        self->abi_extra = PYBIND11_PLATFORM_ABI_ID;
        self->from_python = foreign_cb_from_python;
        self->to_python = foreign_cb_to_python;
        self->keep_alive = foreign_cb_keep_alive;
        self->translate_exception = foreign_cb_translate_exception;
        self->add_foreign_binding = foreign_cb_add_foreign_binding;
        self->remove_foreign_binding = foreign_cb_remove_foreign_binding;
        self->add_foreign_framework = foreign_cb_add_foreign_framework;
        return true;
    });
    if (inited_by_us) {
        // Unlock internals before calling add_framework, so that the callbacks
        // (foreign_cb_add_foreign_binding, etc) can safely re-lock it.
        pymb_add_framework(registry, self.get());
    }
    return inited_by_us;
}

inline foreign_internals::~foreign_internals() = default;

// Learn to satisfy from- and to-Python requests for `cpptype` using the
// foreign binding provided by the given `pytype`. If cpptype is nullptr, infer
// the C++ type by looking at the binding, and require that its ABI match ours.
// Throws an exception on failure. Caller must hold the internals lock and have
// already called foreign_internals.initialize_if_needed().
PYBIND11_NOINLINE void import_foreign_type(type pytype,
                                           const std::type_info *cpptype) {
    auto &foreign_internals = get_foreign_internals();
    pymb_binding* binding = pymb_get_binding(pytype.ptr());
    if (!binding)
        pybind11_fail("pybind11::import_foreign_type(): type does not define "
                      "a __pymetabind_binding__");
    if (binding->framework == foreign_internals.self.get())
        pybind11_fail("pybind11::import_foreign_type(): type is not foreign");
    if (!cpptype) {
        if (binding->framework->abi_lang != pymb_abi_lang_cpp)
            pybind11_fail("pybind11::import_foreign_type(): type is not "
                          "written in C++, so you must specify a C++ type");
        if (binding->framework->abi_extra != foreign_internals.self->abi_extra)
            pybind11_fail("pybind11::import_foreign_type(): type has "
                          "incompatible C++ ABI with this module");
        cpptype = (const std::type_info *) binding->native_type;
    }

    auto result = foreign_internals.manual_imports.emplace(binding, cpptype);
    if (!result.second) {
        auto *existing = (const std::type_info *) result.first->second;
        if (existing != cpptype && *existing != *cpptype)
            pybind11_fail("pybind11::import_foreign_type(): type was "
                          "already imported as a different C++ type");
    }
    import_foreign_binding(binding, cpptype);
}

// Call `import_foreign_binding()` for every ABI-compatible type provided by
// other C++ binding frameworks used by extension modules loaded in this
// interpreter, both those that exist now and those bound in the future.
PYBIND11_NOINLINE void foreign_enable_import_all() {
    auto& foreign_internals = get_foreign_internals();
    bool proceed = with_internals([&](internals&) {
        if (foreign_internals.import_all)
            return false;
        foreign_internals.import_all = true;
        return true;
    });
    if (!proceed)
        return;
    if (foreign_internals.initialize_if_needed()) {
        // pymb_add_framework tells us about every existing type when we
        // register, so if we register with import enabled, we're done
        return;
    }
    // If we enable import after registering, we have to iterate over the
    // list of types ourselves. Do this without the internals lock held so
    // we can reuse the pymb callback functions. foreign_internals registry +
    // self never change once they're non-null, so we can accesss them
    // without locking here.
    pymb_lock_registry(foreign_internals.registry);
    PYMB_LIST_FOREACH(struct pymb_binding*, binding,
                      foreign_internals.registry->bindings) {
        if (binding->framework != foreign_internals.self.get() &&
            pymb_try_ref_binding(binding)) {
            foreign_cb_add_foreign_binding(binding);
            pymb_unref_binding(binding);
        }
    }
    pymb_unlock_registry(foreign_internals.registry);
}

// Expose hooks for other frameworks to use to work with the given pybind11
// type object. Caller must hold the internals lock and have already called
// foreign_internals.initialize_if_needed().
PYBIND11_NOINLINE void export_type_to_foreign(type_info *ti) {
    auto& foreign_internals = get_foreign_internals();
    auto range = foreign_internals.bindings.equal_range(*ti->cpptype);
    for (auto it = range.first; it != range.second; ++it)
        if (it->second->framework == foreign_internals.self.get())
            return; // already exported

    auto *binding = new pymb_binding{};
    binding->framework = foreign_internals.self.get();
    binding->pytype = ti->type;
    binding->native_type = ti->cpptype;
    binding->source_name = strdup(clean_type_id(ti->cpptype->name()).c_str());
    binding->context = ti;

    capsule tie_lifetimes((void *) binding, [](void *p) {
        pymb_binding *binding = (pymb_binding *) p;
        pymb_remove_binding(get_foreign_internals().registry, binding);
        free(const_cast<char*>(binding->source_name));
        delete binding;
    });
    keep_alive_impl((PyObject *) ti->type, tie_lifetimes);

    foreign_internals.bindings.emplace(*ti->cpptype, binding);
    pymb_add_binding(foreign_internals.registry, binding);
}

// Call `export_type_to_foreign()` for each type that currently exists in our
// internals structure and each type created in the future.
PYBIND11_NOINLINE void foreign_enable_export_all() {
    auto& foreign_internals = get_foreign_internals();
    bool proceed = with_internals([&](internals&) {
        if (foreign_internals.export_all)
            return false;
        foreign_internals.export_all = true;
        foreign_internals.export_type_to_foreign =
            &detail::export_type_to_foreign;
        return true;
    });
    if (!proceed)
        return;
    foreign_internals.initialize_if_needed();
    with_internals([&](internals& internals) {
        for (const auto& entry : internals.registered_types_cpp) {
            detail::export_type_to_foreign(entry.second);
        }
    });
}

// Invoke `attempt(closure, binding)` for each foreign binding `binding`
// that claims `type` and was not supplied by us, until one of them returns
// non-null. Return that first non-null value, or null if all attempts failed.
PYBIND11_NOINLINE void* try_foreign_bindings(
        const std::type_info *type,
        void* (*attempt)(void *closure, pymb_binding *binding),
        void *closure) {
    auto &internals = get_internals();
    auto &foreign_internals = get_foreign_internals();

    PYBIND11_LOCK_INTERNALS(internals);
    auto range = foreign_internals.bindings.equal_range(*type);

    if (range.first == range.second)
        return nullptr; // no foreign bindings

    if (std::next(range.first) == range.second) {
        // Single binding - check that it's not our own
        auto *binding = range.first->second;
        if (binding->framework != foreign_internals.self.get() &&
            pymb_try_ref_binding(binding)) {
#ifdef Py_GIL_DISABLED
            // attempt() might execute Python code; drop the internals lock
            // to avoid a deadlock
            lock.unlock();
#endif
            void *result = attempt(closure, binding);
            pymb_unref_binding(binding);
            return result;
        }
        return nullptr;
    }

    // Multiple bindings - try all except our own
#ifndef Py_GIL_DISABLED
    for (auto it = range.first; it != range.second; ++it) {
        auto *binding = it->second;
        if (binding->framework != foreign_internals.self.get() &&
            pymb_try_ref_binding(binding)) {
            void *result = attempt(closure, binding);
            pymb_unref_binding(binding);
            if (result)
                return result;
        }
    }
    return nullptr;
#else
    // In free-threaded mode, this is tricky: we need to drop the
    // internals lock before calling attempt(), but once we do so,
    // any of these bindings that might be in the middle of getting deleted
    // can be concurrently removed from the map, which would interfere
    // with our iteration. Copy the binding pointers out of the list to avoid
    // this problem.

    // Count the number of foreign bindings we might see
    size_t len = (size_t) std::distance(range.first, range.second);

    // Allocate temporary storage for that many pointers
    pymb_binding **scratch =
        (pymb_binding **) alloca(len * sizeof(pymb_binding*));
    pymb_binding **scratch_tail = scratch;

    // Iterate again, taking out strong references and saving pointers to
    // our scratch storage
    for (auto it = range.first; it != range.second; ++it) {
        auto *binding = it->second;
        if (binding->framework != foreign_internals.self.get() &&
            pymb_try_ref_binding(binding))
            *scratch_tail++ = binding;
    }

    // Drop the lock and proceed using only our saved binding pointers.
    // Since we obtained strong references to them, there is no remaining
    // concurrent-destruction hazard.
    lock.unlock();
    void *result = nullptr;
    while (scratch != scratch_tail) {
        if (!result)
            result = attempt(closure, *scratch);
        pymb_unref_binding(*scratch);
        ++scratch;
    }
    return result;
#endif
}

PYBIND11_NAMESPACE_END(detail)

inline void set_foreign_type_defaults(bool export_all, bool import_all) {
    auto &foreign_internals = detail::get_foreign_internals();
    if (import_all && !foreign_internals.import_all)
        detail::foreign_enable_import_all();
    if (export_all && !foreign_internals.export_all)
        detail::foreign_enable_export_all();
}

template <class T = void>
inline void import_foreign_type(type pytype) {
    const std::type_info *cpptype = std::is_void<T>::value ? nullptr : &typeid(T);
    auto& foreign_internals = detail::get_foreign_internals();
    foreign_internals.initialize_if_needed();
    detail::with_internals([&](detail::internals&) {
        detail::import_foreign_type(pytype, cpptype);
    });
}

inline void export_type_to_foreign(type ty) {
    detail::type_info *ti = detail::get_type_info((PyTypeObject *) ty.ptr());
    if (!ti)
        pybind11_fail("pybind11::export_type_to_foreign: not a "
                      "pybind11 registered type");
    auto& foreign_internals = detail::get_foreign_internals();
    foreign_internals.initialize_if_needed();
    detail::with_internals([&](detail::internals&) {
        detail::export_type_to_foreign(ti);
    });
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
