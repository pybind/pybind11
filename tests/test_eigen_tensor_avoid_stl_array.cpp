/*
    tests/eigen_tensor.cpp -- automatic conversion of Eigen Tensor

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

constexpr const char *test_eigen_tensor_module_name = "eigen_tensor_avoid_stl_array";

#ifndef EIGEN_AVOID_STL_ARRAY
#    define EIGEN_AVOID_STL_ARRAY
#endif

#define PYBIND11_TEST_EIGEN_TENSOR_NAMESPACE eigen_tensor_avoid_stl_array

#include "test_eigen_tensor.inl"
