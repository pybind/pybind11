/*
    pybind11/eigen.h: Transparent conversion for dense and sparse Eigen matrices

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "eigen/matrix.h"

#include <Eigen/src/Core/util/Macros.h>

#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#   if __GNUC__ < 5
#       define PYBIND11_CANT_INCLUDE_TENSOR
#   endif
#endif

#if !EIGEN_VERSION_AT_LEAST(3, 3, 0)
#define PYBIND11_CANT_INCLUDE_TENSOR
#endif

#ifndef PYBIND11_CANT_INCLUDE_TENSOR
#    include "eigen/tensor.h"
#endif
