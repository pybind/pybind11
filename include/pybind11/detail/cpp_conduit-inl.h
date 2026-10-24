/*
    pybind11/detail/cpp_conduit-inl.h: Out-of-line definitions for cpp_conduit.h

    Copyright (c) 2024 The pybind Community.
*/

// Every function defined here must start with PYBIND11_INLINE (or
// PYBIND11_NOINLINE_ATTR PYBIND11_INLINE). In the default header-only mode this file is
// included at the bottom of cpp_conduit.h; when PYBIND11_PRECOMPILED is defined it is only
// compiled into the pybind11 static library (see src/).

#pragma once

#include "../conduit/pybind11_platform_abi_id.h"
#include "cpp_conduit.h"

#include <typeinfo>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

PYBIND11_INLINE bool type_is_managed_by_our_internals(PyTypeObject *type_obj) {
#if defined(PYPY_VERSION)
    auto &internals = get_internals();
    return bool(internals.registered_types_py.find(type_obj)
                != internals.registered_types_py.end());
#else
    return (type_obj->tp_new == pybind11_object_new);
#endif
}

PYBIND11_INLINE bool is_instance_method_of_type(PyTypeObject *type_obj, PyObject *attr_name) {
    PyObject *descr = _PyType_Lookup(type_obj, attr_name);
    return ((descr != nullptr) && PyInstanceMethod_Check(descr));
}

PYBIND11_INLINE object try_get_cpp_conduit_method(PyObject *obj) {
    if (PyType_Check(obj)) {
        return object();
    }
    PyTypeObject *type_obj = Py_TYPE(obj);
    str attr_name("_pybind11_conduit_v1_");
    bool assumed_to_be_callable = false;
    if (type_is_managed_by_our_internals(type_obj)) {
        if (!is_instance_method_of_type(type_obj, attr_name.ptr())) {
            return object();
        }
        assumed_to_be_callable = true;
    }
    PyObject *method = PyObject_GetAttr(obj, attr_name.ptr());
    if (method == nullptr) {
        PyErr_Clear();
        return object();
    }
    if (!assumed_to_be_callable && PyCallable_Check(method) == 0) {
        Py_DECREF(method);
        return object();
    }
    return reinterpret_steal<object>(method);
}

PYBIND11_INLINE void *
try_raw_pointer_ephemeral_from_cpp_conduit(handle src, const std::type_info *cpp_type_info) {
    object method = try_get_cpp_conduit_method(src.ptr());
    if (method) {
        capsule cpp_type_info_capsule(const_cast<void *>(static_cast<const void *>(cpp_type_info)),
                                      typeid(std::type_info).name());
        object cpp_conduit = method(bytes(PYBIND11_PLATFORM_ABI_ID),
                                    cpp_type_info_capsule,
                                    bytes("raw_pointer_ephemeral"));
        if (isinstance<capsule>(cpp_conduit)) {
            return reinterpret_borrow<capsule>(cpp_conduit).get_pointer();
        }
    }
    return nullptr;
}

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
