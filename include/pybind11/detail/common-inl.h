/*
    pybind11/detail/common-inl.h -- Basic macros definitions

    Copyright (c) 2016-2022 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/
#include "pybind11/detail/common.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

PYBIND11_NOINLINE_ATTR PYBIND11_INLINE void pybind11_fail(const char *reason) {
    assert(!PyErr_Occurred());
    throw std::runtime_error(reason);
}
PYBIND11_NOINLINE_ATTR PYBIND11_INLINE void pybind11_fail(const std::string &reason) {
    assert(!PyErr_Occurred());
    throw std::runtime_error(reason);
}

PYBIND11_INLINE error_scope::error_scope() { PyErr_Fetch(&type, &value, &trace); }
PYBIND11_INLINE error_scope::~error_scope() { PyErr_Restore(type, value, trace); };

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
