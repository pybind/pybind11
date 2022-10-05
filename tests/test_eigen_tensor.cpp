/*
    tests/eigen_tensor.cpp -- automatic conversion of Eigen Tensor

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
        x = new Eigen::Tensor<double, 3>(3, 5, 2);
        reset_tensor(*x);
    }

    return *x;
}

Eigen::TensorFixedSize<double, Eigen::Sizes<3, 5, 2>> &get_fixed_tensor() {
    static Eigen::TensorFixedSize<double, Eigen::Sizes<3, 5, 2>> *x;

    if (!x) {
        Eigen::aligned_allocator<Eigen::TensorFixedSize<double, Eigen::Sizes<3, 5, 2>>> allocator;
        x = new (allocator.allocate(1)) Eigen::TensorFixedSize<double, Eigen::Sizes<3, 5, 2>>();
        reset_tensor(*x);
    }

    return *x;
}

const Eigen::Tensor<double, 3> &get_const_tensor() { return get_tensor(); }

TEST_SUBMODULE(eigen_tensor, m) {
    m.def(
        "copy_fixed_tensor", []() { return &get_fixed_tensor(); }, py::return_value_policy::copy);

    m.def(
        "copy_tensor", []() { return &get_tensor(); }, py::return_value_policy::copy);

    m.def(
        "copy_const_tensor", []() { return &get_const_tensor(); }, py::return_value_policy::copy);

    m.def("move_fixed_tensor", []() { return get_fixed_tensor(); });

    m.def("move_tensor", []() { return get_tensor(); });

    m.def("move_const_tensor",
          []() -> const Eigen::Tensor<double, 3> { return get_const_tensor(); });

    m.def(
        "take_fixed_tensor",
        []() {
            Eigen::aligned_allocator<Eigen::TensorFixedSize<double, Eigen::Sizes<3, 5, 2>>>
                allocator;
            return new (allocator.allocate(1))
                Eigen::TensorFixedSize<double, Eigen::Sizes<3, 5, 2>>(get_fixed_tensor());
        },
        py::return_value_policy::take_ownership);

    m.def(
        "take_tensor",
        []() { return new Eigen::Tensor<double, 3>(get_tensor()); },
        py::return_value_policy::take_ownership);

    m.def(
        "take_const_tensor",
        []() -> const Eigen::Tensor<double, 3> * {
            return new Eigen::Tensor<double, 3>(get_tensor());
        },
        py::return_value_policy::take_ownership);

    m.def(
        "reference_tensor", []() { return &get_tensor(); }, py::return_value_policy::reference);

    m.def(
        "reference_tensor_v2",
        []() -> Eigen::Tensor<double, 3> & { return get_tensor(); },
        py::return_value_policy::reference);

    m.def(
        "reference_tensor_internal",
        []() { return &get_tensor(); },
        py::return_value_policy::reference_internal);

    m.def(
        "reference_fixed_tensor",
        []() { return &get_tensor(); },
        py::return_value_policy::reference);

    m.def(
        "reference_const_tensor",
        []() { return &get_const_tensor(); },
        py::return_value_policy::reference);

    m.def(
        "reference_const_tensor_v2",
        []() -> const Eigen::Tensor<double, 3> & { return get_const_tensor(); },
        py::return_value_policy::reference);

    m.def(
        "reference_view_of_tensor",
        []() { return Eigen::TensorMap<Eigen::Tensor<double, 3>>(get_tensor()); },
        py::return_value_policy::reference);

    m.def(
        "reference_view_of_fixed_tensor",
        []() {
            return Eigen::TensorMap<Eigen::TensorFixedSize<double, Eigen::Sizes<3, 5, 2>>>(
                get_fixed_tensor());
        },
        py::return_value_policy::reference);

    m.def("round_trip_tensor", [](const Eigen::Tensor<double, 3> &tensor) { return tensor; });

    m.def(
        "round_trip_view_tensor",
        [](Eigen::TensorMap<Eigen::Tensor<double, 3>> view) { return view; },
        py::return_value_policy::reference);

    m.def(
        "round_trip_aligned_view_tensor",
        [](Eigen::TensorMap<Eigen::Tensor<double, 3>, Eigen::Aligned> view) { return view; },
        py::return_value_policy::reference);

    m.def(
        "round_trip_const_view_tensor",
        [](Eigen::TensorMap<Eigen::Tensor<const double, 3>> view) {
            return Eigen::Tensor<double, 3>(view);
        },
        py::return_value_policy::move);
}
