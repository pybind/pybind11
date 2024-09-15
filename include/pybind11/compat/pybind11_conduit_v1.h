// Copyright (c) 2024 The pybind Community.

// THIS MUST STAY AT THE TOP!
#include "pybind11_platform_abi_id.h"

#include <Python.h>
#include <typeinfo>

namespace pybind11_conduit_v1 {

inline void *get_raw_pointer_ephemeral(PyObject *py_obj, const std::type_info *cpp_type_info) {
    PyObject *cpp_type_info_capsule
        = PyCapsule_New(const_cast<void *>(static_cast<const void *>(cpp_type_info)),
                        typeid(std::type_info).name(),
                        nullptr);
    if (cpp_type_info_capsule == nullptr) {
        return nullptr;
    }
    PyObject *cpp_conduit = PyObject_CallMethod(py_obj,
                                                "_pybind11_conduit_v1_",
                                                "yOy",
                                                PYBIND11_PLATFORM_ABI_ID,
                                                cpp_type_info_capsule,
                                                "raw_pointer_ephemeral");
    Py_DECREF(cpp_type_info_capsule);
    if (cpp_conduit == nullptr) {
        return nullptr;
    }
    void *raw_ptr = PyCapsule_GetPointer(cpp_conduit, cpp_type_info->name());
    Py_DECREF(cpp_conduit);
    if (PyErr_Occurred()) {
        return nullptr;
    }
    return raw_ptr;
}

template <typename T>
T *get_type_pointer_ephemeral(PyObject *py_obj) {
    void *raw_ptr = get_raw_pointer_ephemeral(py_obj, &typeid(T));
    if (raw_ptr == nullptr) {
        return nullptr;
    }
    return static_cast<T *>(raw_ptr);
}

} // namespace pybind11_conduit_v1
