// Copyright (c) 2023 The pybind Community.

#pragma once

// Common message for `static_assert()`s, which are useful to easily
// preempt much less obvious errors.
#define PYBIND11_EIGEN_MESSAGE_POINTER_TYPES_ARE_NOT_SUPPORTED                                    \
    "Pointer types (in particular `PyObject *`) are not supported as scalar types for Eigen "     \
    "types."

#define PYBIND11_EIGEN_MESSAGE_MAP_CANNOT_BE_LOADED                                               \
    "Eigen::Map and other map-like types can only be returned to Python, not loaded from "        \
    "Python (as a bound function argument or as a PYBIND11_OVERRIDE return type). Use "           \
    "Eigen::Ref for arguments, or a plain Eigen type (e.g. Eigen::MatrixXd) to get a copy."
