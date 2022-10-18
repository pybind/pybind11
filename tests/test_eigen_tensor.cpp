/*
    tests/eigen_tensor.cpp -- automatic conversion of Eigen Tensor

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

constexpr const char *test_eigen_tensor_module_name = "eigen_tensor";

#define PYBIND11_TEST_EIGEN_TENSOR_NAMESPACE eigen_tensor

#ifdef EIGEN_AVOID_STL_ARRAY
#    undef EIGEN_AVOID_STL_ARRAY
#endif

#include "test_eigen_tensor.inl"
