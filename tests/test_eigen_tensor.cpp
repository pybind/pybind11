/*
    tests/eigen_tensor.cpp -- automatic conversion of Eigen Tensor

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <pybind11/eigen.h>

#include "pybind11_tests.h"

template <typename M>
void reset_tensor(M &x) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 2; k++) {
                x(i, j, k) = i * (5 * 2) + j * 2 + k;
            }
        }
    }
}

template <int Options>
Eigen::Tensor<double, 3, Options> &get_tensor() {
    static Eigen::Tensor<double, 3, Options> *x;

    if (!x) {
        x = new Eigen::Tensor<double, 3, Options>(3, 5, 2);
        reset_tensor(*x);
    }

    return *x;
}

template <int Options>
Eigen::TensorMap<Eigen::Tensor<double, 3, Options>> &get_tensor_map() {
    static Eigen::TensorMap<Eigen::Tensor<double, 3, Options>> *x;

    if (!x) {
        x = new Eigen::TensorMap<Eigen::Tensor<double, 3, Options>>(get_tensor<Options>());
    }

    return *x;
}

template <int Options>
Eigen::TensorFixedSize<double, Eigen::Sizes<3, 5, 2>, Options> &get_fixed_tensor() {
    static Eigen::TensorFixedSize<double, Eigen::Sizes<3, 5, 2>, Options> *x;

    if (!x) {
        Eigen::aligned_allocator<Eigen::TensorFixedSize<double, Eigen::Sizes<3, 5, 2>, Options>>
            allocator;
        x = new (allocator.allocate(1))
            Eigen::TensorFixedSize<double, Eigen::Sizes<3, 5, 2>, Options>();
        reset_tensor(*x);
    }

    return *x;
}

template <int Options>
const Eigen::Tensor<double, 3, Options> &get_const_tensor() {
    return get_tensor<Options>();
}

template <int Options>
struct CustomExample {
    CustomExample() : member(get_tensor<Options>()), view_member(member) {}

    Eigen::Tensor<double, 3, Options> member;
    Eigen::TensorMap<Eigen::Tensor<double, 3, Options>> view_member;
};

template <int Options>
void init_tensor_module(pybind11::module &m) {
    const char *needed_options = "";
    if (PYBIND11_SILENCE_MSVC_C4127(Options == Eigen::ColMajor)) {
        needed_options = "F";
    } else {
        needed_options = "C";
    }
    m.attr("needed_options") = needed_options;

    py::class_<CustomExample<Options>>(m, "CustomExample")
        .def(py::init<>())
        .def_readonly(
            "member", &CustomExample<Options>::member, py::return_value_policy::reference_internal)
        .def_readonly("member_view",
                      &CustomExample<Options>::view_member,
                      py::return_value_policy::reference_internal);

    m.def(
        "copy_fixed_tensor",
        []() { return &get_fixed_tensor<Options>(); },
        py::return_value_policy::copy);

    m.def(
        "copy_tensor", []() { return &get_tensor<Options>(); }, py::return_value_policy::copy);

    m.def(
        "copy_const_tensor",
        []() { return &get_const_tensor<Options>(); },
        py::return_value_policy::copy);

    m.def("move_fixed_tensor", []() { return get_fixed_tensor<Options>(); });

    m.def("move_tensor", []() { return get_tensor<Options>(); });

    m.def("move_const_tensor",
          // NOLINTNEXTLINE(readability-const-return-type)
          []() -> const Eigen::Tensor<double, 3, Options> { return get_const_tensor<Options>(); });

    m.def(
        "take_fixed_tensor",

        []() {
            Eigen::aligned_allocator<
                Eigen::TensorFixedSize<double, Eigen::Sizes<3, 5, 2>, Options>>
                allocator;
            return new (allocator.allocate(1))
                Eigen::TensorFixedSize<double, Eigen::Sizes<3, 5, 2>, Options>(
                    get_fixed_tensor<Options>());
        },
        py::return_value_policy::take_ownership);

    m.def(
        "take_tensor",
        []() { return new Eigen::Tensor<double, 3, Options>(get_tensor<Options>()); },
        py::return_value_policy::take_ownership);

    m.def(
        "take_const_tensor",
        []() -> const Eigen::Tensor<double, 3, Options> * {
            return new Eigen::Tensor<double, 3, Options>(get_tensor<Options>());
        },
        py::return_value_policy::take_ownership);

    m.def(
        "reference_tensor",
        []() { return &get_tensor<Options>(); },
        py::return_value_policy::reference);

    m.def(
        "reference_tensor_v2",
        []() -> Eigen::Tensor<double, 3, Options> & { return get_tensor<Options>(); },
        py::return_value_policy::reference);

    m.def(
        "reference_tensor_internal",
        []() { return &get_tensor<Options>(); },
        py::return_value_policy::reference_internal);

    m.def(
        "reference_fixed_tensor",
        []() { return &get_tensor<Options>(); },
        py::return_value_policy::reference);

    m.def(
        "reference_const_tensor",
        []() { return &get_const_tensor<Options>(); },
        py::return_value_policy::reference);

    m.def(
        "reference_const_tensor_v2",
        []() -> const Eigen::Tensor<double, 3, Options> & { return get_const_tensor<Options>(); },
        py::return_value_policy::reference);

    m.def(
        "reference_view_of_tensor",
        []() -> Eigen::TensorMap<Eigen::Tensor<double, 3, Options>> {
            return get_tensor_map<Options>();
        },
        py::return_value_policy::reference);

    m.def(
        "reference_view_of_tensor_v2",
        // NOLINTNEXTLINE(readability-const-return-type)
        []() -> const Eigen::TensorMap<Eigen::Tensor<double, 3, Options>> {
            return get_tensor_map<Options>(); // NOLINT(readability-const-return-type)
        },                                    // NOLINT(readability-const-return-type)
        py::return_value_policy::reference);

    m.def(
        "reference_view_of_tensor_v3",
        []() -> Eigen::TensorMap<Eigen::Tensor<double, 3, Options>> * {
            return &get_tensor_map<Options>();
        },
        py::return_value_policy::reference);

    m.def(
        "reference_view_of_tensor_v4",
        []() -> const Eigen::TensorMap<Eigen::Tensor<double, 3, Options>> * {
            return &get_tensor_map<Options>();
        },
        py::return_value_policy::reference);

    m.def(
        "reference_view_of_tensor_v5",
        []() -> Eigen::TensorMap<Eigen::Tensor<double, 3, Options>> & {
            return get_tensor_map<Options>();
        },
        py::return_value_policy::reference);

    m.def(
        "reference_view_of_tensor_v6",
        []() -> const Eigen::TensorMap<Eigen::Tensor<double, 3, Options>> & {
            return get_tensor_map<Options>();
        },
        py::return_value_policy::reference);

    m.def(
        "reference_view_of_fixed_tensor",
        []() {
            return Eigen::TensorMap<
                Eigen::TensorFixedSize<double, Eigen::Sizes<3, 5, 2>, Options>>(
                get_fixed_tensor<Options>());
        },
        py::return_value_policy::reference);

    m.def("round_trip_tensor",
          [](const Eigen::Tensor<double, 3, Options> &tensor) { return tensor; });

    m.def("round_trip_fixed_tensor",
          [](const Eigen::TensorFixedSize<double, Eigen::Sizes<3, 5, 2>, Options> &tensor) { return tensor; });

    m.def(
        "round_trip_view_tensor",
        [](Eigen::TensorMap<Eigen::Tensor<double, 3, Options>> view) { return view; },
        py::return_value_policy::reference);

    m.def(
        "round_trip_view_tensor_ref",
        [](Eigen::TensorMap<Eigen::Tensor<double, 3, Options>> &view) { return view; },
        py::return_value_policy::reference);

    m.def(
        "round_trip_view_tensor_ptr",
        [](Eigen::TensorMap<Eigen::Tensor<double, 3, Options>> *view) { return view; },
        py::return_value_policy::reference);

    m.def(
        "round_trip_aligned_view_tensor",
        [](Eigen::TensorMap<Eigen::Tensor<double, 3, Options>, Eigen::Aligned> view) {
            return view;
        },
        py::return_value_policy::reference);

    m.def(
        "round_trip_const_view_tensor",
        [](Eigen::TensorMap<Eigen::Tensor<const double, 3, Options>> view) {
            return Eigen::Tensor<double, 3, Options>(view);
        },
        py::return_value_policy::move);
}

TEST_SUBMODULE(eigen_tensor, m) {
    auto f_style = m.def_submodule("f_style");
    auto c_style = m.def_submodule("c_style");

    init_tensor_module<Eigen::ColMajor>(f_style);
    init_tensor_module<Eigen::RowMajor>(c_style);
}
