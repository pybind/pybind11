/*
    tests/eigen_tensor.cpp -- automatic conversion of Eigen Tensor

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <pybind11/eigen.h>

#include "pybind11_tests.h"

template <typename M>
void reset_tensor(M &x) {
    for (int i = 0; i < x.size(); i++) {
        x(i) = i;
    }
}

Eigen::Tensor<double, 3> &get_tensor() {
    static Eigen::Tensor<double, 3> *x;

    if (!x) {
        x = new Eigen::Tensor<double, 3>(3, 1, 2);
        reset_tensor(*x);
    }

    return *x;
}

Eigen::TensorFixedSize<double, Eigen::Sizes<3, 1, 2>> &get_fixed_tensor() {
    static Eigen::TensorFixedSize<double, Eigen::Sizes<3, 1, 2>> *x;

    if (!x) {
        Eigen::aligned_allocator<Eigen::TensorFixedSize<double, Eigen::Sizes<3, 1, 2>>> allocator;
        x = new (allocator.allocate(1)) Eigen::TensorFixedSize<double, Eigen::Sizes<3, 1, 2>>();
        reset_tensor(*x);
    }

    return *x;
}

const Eigen::Tensor<double, 3> &get_const_tensor() { return get_tensor(); }

TEST_SUBMODULE(eigen_tensor, m) {
    m.def("copy_fixed_global_tensor", []() { return get_fixed_tensor(); });

    m.def("copy_global_tensor", []() { return get_tensor(); });

    m.def("copy_const_global_tensor", []() { return get_const_tensor(); });

    m.def(
        "reference_global_tensor",
        []() { return &get_tensor(); },
        py::return_value_policy::reference);

    m.def(
        "reference_const_global_tensor",
        []() { return &get_const_tensor(); },
        py::return_value_policy::reference);

    m.def(
        "reference_view_of_global_tensor",
        []() { return Eigen::TensorMap<Eigen::Tensor<double, 3>>(get_tensor()); },
        py::return_value_policy::reference);

    m.def(
        "reference_view_of_fixed_global_tensor",
        []() {
            return Eigen::TensorMap<Eigen::TensorFixedSize<double, Eigen::Sizes<3, 1, 2>>>(
                get_fixed_tensor());
        },
        py::return_value_policy::reference);

    m.def("round_trip_tensor", [](const Eigen::Tensor<double, 3> &tensor) {
        return tensor;
    });

    m.def(
        "round_trip_view_tensor",
        [](Eigen::TensorMap<Eigen::Tensor<double, 3>> view) {
            return view;
        },
        py::return_value_policy::reference);
}
