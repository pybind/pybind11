/*
    tests/test_docstring_options.cpp -- generation of docstrings and signatures

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

struct DocstringTestFoo {
    int value;
    void setValue(int v) { value = v; }
    int getValue() const { return value; }
};

test_initializer docstring_generation([](py::module &m) {

    {
        py::options options;
        options.disable_function_signatures();

        m.def("test_function1", [](int, int) {}, py::arg("a"), py::arg("b"));
        m.def("test_function2", [](int, int) {}, py::arg("a"), py::arg("b"), "A custom docstring");

        options.enable_function_signatures();

        m.def("test_function3", [](int, int) {}, py::arg("a"), py::arg("b"));
        m.def("test_function4", [](int, int) {}, py::arg("a"), py::arg("b"), "A custom docstring");

        options.disable_function_signatures().disable_user_defined_docstrings();

        m.def("test_function5", [](int, int) {}, py::arg("a"), py::arg("b"), "A custom docstring");

        {
            py::options nested_options;
            nested_options.enable_user_defined_docstrings();
            m.def("test_function6", [](int, int) {}, py::arg("a"), py::arg("b"), "A custom docstring");
        }
    }

    m.def("test_function7", [](int, int) {}, py::arg("a"), py::arg("b"), "A custom docstring");

    {
        py::options options;
        options.disable_user_defined_docstrings();

        py::class_<DocstringTestFoo>(m, "DocstringTestFoo", "This is a class docstring")
            .def_property("value_prop", &DocstringTestFoo::getValue, &DocstringTestFoo::setValue, "This is a property docstring")
        ;
    }
});
