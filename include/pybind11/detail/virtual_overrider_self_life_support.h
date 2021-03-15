// Copyright (c) 2021 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include "common.h"
#include "smart_holder_poc.h"
#include "type_caster_base.h"

#include <type_traits>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

// SMART_HOLDER_WIP: Needs refactoring of existing pybind11 code.
inline bool deregister_instance(instance *self, void *valptr, const type_info *tinfo);

// The original core idea for this struct goes back to PyCLIF:
// https://github.com/google/clif/blob/07f95d7e69dca2fcf7022978a55ef3acff506c19/clif/python/runtime.cc#L37
// URL provided here mainly to give proper credit. To fully explain the `HoldPyObj` feature, more
// context is needed (SMART_HOLDER_WIP).
struct virtual_overrider_self_life_support {
    value_and_holder loaded_v_h;
    ~virtual_overrider_self_life_support() {
        if (loaded_v_h.inst != nullptr && loaded_v_h.vh != nullptr) {
            void *value_void_ptr = loaded_v_h.value_ptr();
            if (value_void_ptr != nullptr) {
                PyGILState_STATE threadstate = PyGILState_Ensure();
                Py_DECREF((PyObject *) loaded_v_h.inst);
                loaded_v_h.value_ptr() = nullptr;
                loaded_v_h.holder<pybindit::memory::smart_holder>().release_disowned();
                deregister_instance(loaded_v_h.inst, value_void_ptr, loaded_v_h.type);
                PyGILState_Release(threadstate);
            }
        }
    }

    // Some compilers complain about implicitly defined versions of some of the following:
    virtual_overrider_self_life_support()                                            = default;
    virtual_overrider_self_life_support(const virtual_overrider_self_life_support &) = default;
    virtual_overrider_self_life_support(virtual_overrider_self_life_support &&)      = default;
    virtual_overrider_self_life_support &operator=(const virtual_overrider_self_life_support &)
        = default;
    virtual_overrider_self_life_support &operator=(virtual_overrider_self_life_support &&)
        = default;
};

template <typename T, detail::enable_if_t<!std::is_polymorphic<T>::value, int> = 0>
virtual_overrider_self_life_support *
dynamic_cast_virtual_overrider_self_life_support_ptr(T * /*raw_type_ptr*/) {
    return nullptr;
}

template <typename T, detail::enable_if_t<std::is_polymorphic<T>::value, int> = 0>
virtual_overrider_self_life_support *
dynamic_cast_virtual_overrider_self_life_support_ptr(T *raw_type_ptr) {
    return dynamic_cast<virtual_overrider_self_life_support *>(raw_type_ptr);
}

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
