// Copyright (c) 2021 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include "detail/common.h"
#include "detail/smart_holder_poc.h"
#include "detail/type_caster_base.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

PYBIND11_NAMESPACE_BEGIN(detail)
// SMART_HOLDER_WIP: Needs refactoring of existing pybind11 code.
inline bool deregister_instance(instance *self, void *valptr, const type_info *tinfo);
PYBIND11_NAMESPACE_END(detail)

// The original core idea for this struct goes back to PyCLIF:
// https://github.com/google/clif/blob/07f95d7e69dca2fcf7022978a55ef3acff506c19/clif/python/runtime.cc#L37
// URL provided here mainly to give proper credit. To fully explain the `HoldPyObj` feature, more
// context is needed (SMART_HOLDER_WIP).
struct trampoline_self_life_support {
    detail::value_and_holder v_h;

    trampoline_self_life_support() = default;

    void activate_life_support(const detail::value_and_holder &v_h_) {
        Py_INCREF((PyObject *) v_h_.inst);
        v_h = v_h_;
    }

    void deactivate_life_support() {
        Py_DECREF((PyObject *) v_h.inst);
        v_h = detail::value_and_holder();
    }

    ~trampoline_self_life_support() {
        if (v_h.inst != nullptr && v_h.vh != nullptr) {
            void *value_void_ptr = v_h.value_ptr();
            if (value_void_ptr != nullptr) {
                PyGILState_STATE threadstate = PyGILState_Ensure();
                v_h.value_ptr() = nullptr;
                v_h.holder<pybindit::memory::smart_holder>().release_disowned();
                detail::deregister_instance(v_h.inst, value_void_ptr, v_h.type);
                Py_DECREF((PyObject *) v_h.inst); // Must be after deregister.
                PyGILState_Release(threadstate);
            }
        }
    }

    // For the next two, the default implementations generate undefined behavior (ASAN failures
    // manually verified). The reason is that v_h needs to be kept default-initialized.
    trampoline_self_life_support(const trampoline_self_life_support &) {}
    trampoline_self_life_support(trampoline_self_life_support &&) noexcept {}

    // These should never be needed (please provide test cases if you think they are).
    trampoline_self_life_support &operator=(const trampoline_self_life_support &) = delete;
    trampoline_self_life_support &operator=(trampoline_self_life_support &&) = delete;
};

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
