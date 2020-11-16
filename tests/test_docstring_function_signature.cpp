/*
    tests/test_docstring_options.cpp -- generation of docstrings function signatures

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include "pybind11/stl.h"

enum class Color {Red};

TEST_SUBMODULE(docstring_function_signature, m) {
    // test_docstring_function_signatures
    pybind11::enum_<Color> (m, "Color").value("Red", Color::Red);
    m.def("a", [](Color) {}, pybind11::arg("a") = Color::Red);
    m.def("b", [](int) {}, pybind11::arg("a") = 1);
    m.def("c", [](std::vector<int>) {}, pybind11::arg("a") = std::vector<int> {{1, 2, 3, 4}});
    m.def("d", [](UserType) {}, pybind11::arg("a") = UserType {});
    m.def("e", [](std::pair<UserType, int>) {}, pybind11::arg("a") = std::make_pair<UserType, int>(UserType(), 4));
    m.def("f", [](std::vector<Color>) {}, pybind11::arg("a") = std::vector<Color> {Color::Red});
    m.def("g", [](std::tuple<int, Color, double>) {}, pybind11::arg("a") = std::make_tuple(4, Color::Red, 1.9));
}
