// Copyright (c) 2022 The pybind Community.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include "common.h"

#if defined(WITH_THREAD) && defined(PYBIND11_SIMPLE_GIL_MANAGEMENT)
#    include "../gil.h"
#endif

#include "../pytypes.h"
#include "abi_platform_id.h"

#include <string>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

inline object get_python_state_dict() {
    object state_dict;
#if (PYBIND11_INTERNALS_VERSION <= 4 && PY_VERSION_HEX < 0x030C0000)                              \
    || PY_VERSION_HEX < 0x03080000 || defined(PYPY_VERSION)
    state_dict = reinterpret_borrow<object>(PyEval_GetBuiltins());
#else
#    if PY_VERSION_HEX < 0x03090000
    PyInterpreterState *istate = _PyInterpreterState_Get();
#    else
    PyInterpreterState *istate = PyInterpreterState_Get();
#    endif
    if (istate) {
        state_dict = reinterpret_borrow<object>(PyInterpreterState_GetDict(istate));
    }
#endif
    if (!state_dict) {
        raise_from(PyExc_SystemError, "pybind11::detail::get_python_state_dict() FAILED");
    }
    return state_dict;
}

#if defined(WITH_THREAD)
#    if defined(PYBIND11_SIMPLE_GIL_MANAGEMENT)
using gil_scoped_acquire_simple = gil_scoped_acquire;
#    else
// Cannot use py::gil_scoped_acquire here since that constructor calls get_internals.
struct gil_scoped_acquire_simple {
    gil_scoped_acquire_simple() : state(PyGILState_Ensure()) {}
    gil_scoped_acquire_simple(const gil_scoped_acquire_simple &) = delete;
    gil_scoped_acquire_simple &operator=(const gil_scoped_acquire_simple &) = delete;
    ~gil_scoped_acquire_simple() { PyGILState_Release(state); }
    const PyGILState_STATE state;
};
#    endif
#endif

/* NOTE: struct cross_extension_shared_state is in
             namespace pybind11::detail
         but all types using this struct are meant to live in
             namespace pybind11::cross_extension_shared_states
         to make them easy to discover and reason about.
 */
template <typename AdapterType>
struct cross_extension_shared_state {
    static constexpr const char *abi_id() { return AdapterType::abi_id(); }

    using payload_type = typename AdapterType::payload_type;

    static payload_type **&payload_pp() {
        // The reason for the double-indirection is documented here:
        // https://github.com/pybind/pybind11/pull/1092
        static payload_type **pp;
        return pp;
    }

    static payload_type *get_existing() {
        if (payload_pp() && *payload_pp()) {
            return *payload_pp();
        }

        gil_scoped_acquire_simple gil;
        error_scope err_scope;

        str abi_id_str(AdapterType::abi_id());
        dict state_dict = get_python_state_dict();
        if (!state_dict.contains(abi_id_str)) {
            return nullptr;
        }

        void *raw_ptr = PyCapsule_GetPointer(state_dict[abi_id_str].ptr(), AdapterType::abi_id());
        if (raw_ptr == nullptr) {
            raise_from(PyExc_SystemError,
                       ("pybind11::detail::cross_extension_shared_state::get_existing():"
                        " Retrieve payload_type** from capsule FAILED for ABI ID \""
                        + std::string(AdapterType::abi_id()) + "\"")
                           .c_str());
        }
        payload_pp() = static_cast<payload_type **>(raw_ptr);
        return *payload_pp();
    }

    static payload_type &get() {
        payload_type *existing = get_existing();
        if (existing != nullptr) {
            return *existing;
        }
        if (payload_pp() == nullptr) {
            payload_pp() = new payload_type *();
        }
        *payload_pp() = new payload_type();
        get_python_state_dict()[AdapterType::abi_id()]
            = capsule(payload_pp(), AdapterType::abi_id());
        return **payload_pp();
    }

    struct scoped_clear {
        // To be called BEFORE Py_Finalize().
        scoped_clear() {
            payload_type *existing = get_existing();
            if (existing != nullptr) {
                AdapterType::payload_clear(*existing);
                arm_dtor = true;
            }
        }

        // To be called AFTER Py_Finalize().
        ~scoped_clear() {
            if (arm_dtor) {
                delete *payload_pp();
                *payload_pp() = nullptr;
            }
        }

        scoped_clear(const scoped_clear &) = delete;
        scoped_clear &operator=(const scoped_clear &) = delete;

        bool arm_dtor = false;
    };
};

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
