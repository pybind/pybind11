/*
    tests/test_docstring_options.cpp -- generation of docstrings and signatures

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"



test_initializer docstring_generation([](py::module &m) {

    py::docstring_options docstrings(true, false);  // Enable custom docstrings, disable auto-generated signatures

    m.def("test_function1", [](int, int) {}, py::arg("a"), py::arg("b"));
    m.def("test_function2", [](int, int) {}, py::arg("a"), py::arg("b"), "A custom docstring");

    docstrings.enable_signatures();

    m.def("test_function3", [](int, int) {}, py::arg("a"), py::arg("b"));
    m.def("test_function4", [](int, int) {}, py::arg("a"), py::arg("b"), "A custom docstring");
});
