/*
    pybind11/eigen.h: Transparent conversion for dense and sparse Eigen matrices

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "detail/eigen_matrix.h"
#include "detail/eigen_tensor.h"

// Eigen prior to 3.2.7 doesn't have proper move constructors--but worse, some classes get implicit
// move constructors that break things.  We could detect this an explicitly copy, but an extra copy
// of matrices seems highly undesirable.
static_assert(EIGEN_VERSION_AT_LEAST(3, 2, 7),
              "Eigen support in pybind11 requires Eigen >= 3.2.7");
