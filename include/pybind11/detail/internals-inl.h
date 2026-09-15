/*
    pybind11/detail/internals-inl.h: Out-of-line definitions for internals.h

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

// Every function defined here must start with PYBIND11_INLINE (or
// PYBIND11_NOINLINE_ATTR PYBIND11_INLINE). In the default header-only mode this file is
// included at the bottom of internals.h; when PYBIND11_PRECOMPILED is defined it is only
// compiled into the pybind11 static library (see src/).

#pragma once

#include "internals.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

PYBIND11_INLINE object get_python_state_dict() {
    object state_dict;
#if defined(PYPY_VERSION) || defined(GRAALVM_PYTHON)
    state_dict = reinterpret_borrow<object>(PyEval_GetBuiltins());
#else
    auto *istate = get_interpreter_state_unchecked();
    if (istate) {
        state_dict = reinterpret_borrow<object>(PyInterpreterState_GetDict(istate));
    }
#endif
    if (!state_dict) {
        raise_from(PyExc_SystemError, "pybind11::detail::get_python_state_dict() FAILED");
        throw error_already_set();
    }
    return state_dict;
}

PYBIND11_INLINE uint64_t round_up_to_next_pow2(uint64_t x) {
    // Round-up to the next power of two.
    // See https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
    x--;
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    x |= (x >> 32);
    x++;
    return x;
}

PYBIND11_INLINE std::atomic_bool &has_seen_non_main_interpreter() {
    static std::atomic_bool multi(false);
    return multi;
}

PYBIND11_INLINE bool raise_err(PyObject *exc_type, const char *msg) {
    if (PyErr_Occurred()) {
        raise_from(exc_type, msg);
        return true;
    }
    set_error(exc_type, msg);
    return false;
}

PYBIND11_INLINE void translate_exception(std::exception_ptr p) {
    if (!p) {
        return;
    }
    try {
        std::rethrow_exception(p);
    } catch (error_already_set &e) {
        handle_nested_exception(e, p);
        e.restore();
        return;
    } catch (const builtin_exception &e) {
        // Could not use template since it's an abstract class.
        if (const auto *nep = dynamic_cast<const std::nested_exception *>(std::addressof(e))) {
            handle_nested_exception(*nep, p);
        }
        e.set_error();
        return;
    } catch (const std::bad_alloc &e) {
        handle_nested_exception(e, p);
        raise_err(PyExc_MemoryError, e.what());
        return;
    } catch (const std::domain_error &e) {
        handle_nested_exception(e, p);
        raise_err(PyExc_ValueError, e.what());
        return;
    } catch (const std::invalid_argument &e) {
        handle_nested_exception(e, p);
        raise_err(PyExc_ValueError, e.what());
        return;
    } catch (const std::length_error &e) {
        handle_nested_exception(e, p);
        raise_err(PyExc_ValueError, e.what());
        return;
    } catch (const std::out_of_range &e) {
        handle_nested_exception(e, p);
        raise_err(PyExc_IndexError, e.what());
        return;
    } catch (const std::range_error &e) {
        handle_nested_exception(e, p);
        raise_err(PyExc_ValueError, e.what());
        return;
    } catch (const std::overflow_error &e) {
        handle_nested_exception(e, p);
        raise_err(PyExc_OverflowError, e.what());
        return;
    } catch (const std::exception &e) {
        handle_nested_exception(e, p);
        raise_err(PyExc_RuntimeError, e.what());
        return;
    } catch (const std::nested_exception &e) {
        handle_nested_exception(e, p);
        raise_err(PyExc_RuntimeError, "Caught an unknown nested exception!");
        return;
    } catch (...) {
        raise_err(PyExc_RuntimeError, "Caught an unknown exception!");
        return;
    }
}

// Only declared (and used) on non-libstdc++ platforms; see the comment on the
// declaration in internals.h. Match the guard so precompiled builds do not
// emit undeclared external-linkage definitions.
#if !defined(__GLIBCXX__)
PYBIND11_INLINE void translate_local_exception(std::exception_ptr p) {
    try {
        if (p) {
            std::rethrow_exception(p);
        }
    } catch (error_already_set &e) {
        e.restore();
        return;
    } catch (const builtin_exception &e) {
        e.set_error();
        return;
    }
}

PYBIND11_INLINE void check_internals_local_exception_translator(internals *internals_ptr) {
    if (internals_ptr) {
        for (auto et : internals_ptr->registered_exception_translators) {
            if (et == &translate_local_exception) {
                return;
            }
        }
        internals_ptr->registered_exception_translators.push_front(&translate_local_exception);
    }
}
#endif

PYBIND11_INLINE internals_pp_manager<internals> &get_internals_pp_manager() {
#if defined(__GLIBCXX__)
#    define ON_FETCH_FN nullptr
#else
#    define ON_FETCH_FN &check_internals_local_exception_translator
#endif
    return internals_pp_manager<internals>::get_instance(PYBIND11_INTERNALS_ID, ON_FETCH_FN);
#undef ON_FETCH_FN
}

PYBIND11_NOINLINE_ATTR PYBIND11_INLINE internals &get_internals() {
    auto &ppmgr = get_internals_pp_manager();
    auto *pp = ppmgr.get_pp();
    if (!pp) {
        pybind11_fail("get_internals: get_pp() returned nullptr");
    }
    auto &internals_ptr = *pp;
    if (!internals_ptr) {
        // Slow path, something needs fetched from the state dict or created
        gil_scoped_acquire_simple gil;
        error_scope err_scope;

        ppmgr.create_pp_content_once(&internals_ptr);

        if (!internals_ptr) {
            pybind11_fail("get_internals: create_pp_content_once() produced nullptr");
        }
        if (!internals_ptr->instance_base) {
            // This calls get_internals, so cannot be called from within the internals constructor
            // called above because internals_ptr must be set before get_internals is called again
            internals_ptr->instance_base = make_object_base_type(internals_ptr->default_metaclass);
        }
    }
    return *internals_ptr;
}

PYBIND11_INLINE PyObject *get_internals_capsule() {
    auto state_dict = reinterpret_borrow<dict>(get_python_state_dict());
    return dict_getitemstring(state_dict.ptr(), PYBIND11_INTERNALS_ID);
}

PYBIND11_INLINE const std::string &get_local_internals_key() {
    static const std::string key
        = PYBIND11_MODULE_LOCAL_ID + std::to_string(reinterpret_cast<uintptr_t>(&key));
    return key;
}

PYBIND11_INLINE PyObject *get_local_internals_capsule() {
    const auto &key = get_local_internals_key();
    auto state_dict = reinterpret_borrow<dict>(get_python_state_dict());
    return dict_getitemstring(state_dict.ptr(), key.c_str());
}

PYBIND11_INLINE void ensure_internals() {
    pybind11::detail::get_internals_pp_manager().unref();
#ifdef PYBIND11_HAS_SUBINTERPRETER_SUPPORT
    if (PyInterpreterState_Get() != PyInterpreterState_Main()) {
        has_seen_non_main_interpreter() = true;
    }
#endif
    pybind11::detail::get_internals();
}

PYBIND11_INLINE internals_pp_manager<local_internals> &get_local_internals_pp_manager() {
    // Use the address of a static variable as part of the key, so that the value is uniquely tied
    // to where the module is loaded in memory
    return internals_pp_manager<local_internals>::get_instance(get_local_internals_key().c_str(),
                                                               nullptr);
}

PYBIND11_INLINE local_internals &get_local_internals() {
    auto &ppmgr = get_local_internals_pp_manager();
    auto &internals_ptr = *ppmgr.get_pp();
    if (!internals_ptr) {
        gil_scoped_acquire_simple gil;
        error_scope err_scope;

        ppmgr.create_pp_content_once(&internals_ptr);
    }
    return *internals_ptr;
}

PYBIND11_INLINE size_t num_registered_instances() {
    auto &internals = get_internals();
#ifdef Py_GIL_DISABLED
    size_t count = 0;
    for (size_t i = 0; i <= internals.instance_shards_mask; ++i) {
        auto &shard = internals.instance_shards[i];
        std::unique_lock<pymutex> lock(shard.mutex);
        count += shard.registered_instances.size();
    }
    return count;
#else
    return internals.registered_instances.size();
#endif
}

#if defined(PYBIND11_PRECOMPILED)
// Link-time configuration guard; see the declaration in internals.h.
PYBIND11_INLINE void PYBIND11_PRECOMPILED_CONFIG_CHECK() {}
#endif

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
