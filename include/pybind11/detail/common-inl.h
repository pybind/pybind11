/*
    pybind11/detail/common-inl.h: Out-of-line definitions for common.h

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

// Every function defined here must start with PYBIND11_INLINE (or
// PYBIND11_NOINLINE_ATTR PYBIND11_INLINE). In the default header-only mode this file is
// included at the bottom of common.h; when PYBIND11_PRECOMPILED is defined it is only
// compiled into the pybind11 static library (see src/).

#pragma once

#include "common.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

[[noreturn]] PYBIND11_NOINLINE_ATTR PYBIND11_INLINE void pybind11_fail(const char *reason) {
    assert(!PyErr_Occurred());
    throw std::runtime_error(reason);
}

[[noreturn]] PYBIND11_NOINLINE_ATTR PYBIND11_INLINE void pybind11_fail(const std::string &reason) {
    assert(!PyErr_Occurred());
    throw std::runtime_error(reason);
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
