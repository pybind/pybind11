/*
    pybind11/warnings.h: Python warnings wrappers.

    Copyright (c) 2024 Jan Iwaszkiewicz <jiwaszkiewicz6@gmail.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "pybind11.h"
#include "detail/common.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

PYBIND11_NAMESPACE_BEGIN(detail)

inline bool PyWarning_Check(PyObject *obj) {
    int result = PyObject_IsSubclass(obj, PyExc_Warning);
    if (result == 1) {
        return true;
    }
    if (result == -1) {
        raise_from(PyExc_SystemError,
                   "PyWarning_Check(): internal error of Python C API while "
                   "checking a subclass of the object!");
        throw error_already_set();
    }
    return false;
}

PYBIND11_NAMESPACE_END(detail)

PYBIND11_NAMESPACE_BEGIN(warnings)

inline object
new_warning_type(handle scope, const char *name, handle base = PyExc_RuntimeWarning) {
    if (!detail::PyWarning_Check(base.ptr())) {
        pybind11_fail("warning(): cannot create custom warning, base must be a subclass of "
                      "PyExc_Warning!");
    }
    if (hasattr(scope, "__dict__") && scope.attr("__dict__").contains(name)) {
        pybind11_fail("Error during initialization: multiple incompatible "
                      "definitions with name \""
                      + std::string(name) + "\"");
    }
    std::string full_name = scope.attr("__name__").cast<std::string>() + std::string(".") + name;
    handle h(PyErr_NewException(const_cast<char *>(full_name.c_str()), base.ptr(), nullptr));
    object obj = reinterpret_steal<object>(h);
    scope.attr(name) = obj;
    return obj;
}

// Similar to Python `warnings.warn()`
inline void
warn(const char *message, handle category = PyExc_RuntimeWarning, int stack_level = 2) {
    if (!detail::PyWarning_Check(category.ptr())) {
        pybind11_fail("raise_warning(): cannot raise warning, category must be a subclass of "
                      "PyExc_Warning!");
    }

    if (PyErr_WarnEx(category.ptr(), message, stack_level) == -1) {
        throw error_already_set();
    }
}

PYBIND11_NAMESPACE_END(warnings)

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
