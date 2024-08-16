/*
    tests/test_docstring_options.cpp -- generation of docstrings and signatures

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

TEST_SUBMODULE(docstring_options, m) {
    // test_docstring_options
    {
        py::options options;
        options.disable_function_signatures();

        m.def("test_function1", [](int, int) {}, py::arg("a"), py::arg("b"));
        m.def("test_function2", [](int, int) {}, py::arg("a"), py::arg("b"), "A custom docstring");

        m.def("test_overloaded1", [](int) {}, py::arg("i"), "Overload docstring");
        m.def("test_overloaded1", [](double) {}, py::arg("d"));

        m.def("test_overloaded2", [](int) {}, py::arg("i"), "overload docstring 1");
        m.def("test_overloaded2", [](double) {}, py::arg("d"), "overload docstring 2");

        m.def("test_overloaded3", [](int) {}, py::arg("i"));
        m.def("test_overloaded3", [](double) {}, py::arg("d"), "Overload docstr");

        m.def(
            "test_overloaded4",
            [](int a, int b) -> int { return a + b; },
            "A function which adds two numbers.\n");
        m.def(
            "test_overloaded4",
            [](float a, float b) -> float { return a + b; },
            "Internally, a simple addition is performed.");
        m.def(
            "test_overloaded4",
            [](const py::none &, const py::none &) -> py::none { return py::none(); },
            "Both numbers can be None, and None will be returned.");

        options.enable_function_signatures();

        m.def(
            "test_overloaded5",
            [](int a, int b) -> int { return a + b; },
            "Add two integers together.");
        m.def(
            "test_overloaded5",
            [](float a, float b) -> float { return a + b; },
            "Add two floating point numbers together.");

        m.def("test_function3", [](int, int) {}, py::arg("a"), py::arg("b"));
        m.def("test_function4", [](int, int) {}, py::arg("a"), py::arg("b"), "A custom docstring");

        options.disable_function_signatures().disable_user_defined_docstrings();

        m.def("test_function5", [](int, int) {}, py::arg("a"), py::arg("b"), "A custom docstring");

        {
            py::options nested_options;
            nested_options.enable_user_defined_docstrings();
            m.def(
                "test_function6",
                [](int, int) {},
                py::arg("a"),
                py::arg("b"),
                "A custom docstring");
        }
    }

    m.def("test_function7", [](int, int) {}, py::arg("a"), py::arg("b"), "A custom docstring");

    {
        py::options options;
        options.disable_user_defined_docstrings();
        options.disable_function_signatures();

        m.def("test_function8", []() {});
    }

    {
        py::options options;
        options.disable_user_defined_docstrings();

        struct DocstringTestFoo {
            int value;
            void setValue(int v) { value = v; }
            int getValue() const { return value; }
        };
        py::class_<DocstringTestFoo>(m, "DocstringTestFoo", "This is a class docstring")
            .def_property("value_prop",
                          &DocstringTestFoo::getValue,
                          &DocstringTestFoo::setValue,
                          "This is a property docstring");
    }

    {
        enum class DocstringTestEnum1 { Member1, Member2 };

        py::enum_<DocstringTestEnum1>(m, "DocstringTestEnum1", "Enum docstring")
            .value("Member1", DocstringTestEnum1::Member1)
            .value("Member2", DocstringTestEnum1::Member2);
    }

    {
        py::options options;
        options.enable_enum_members_docstring();

        enum class DocstringTestEnum2 { Member1, Member2 };

        py::enum_<DocstringTestEnum2>(m, "DocstringTestEnum2", "Enum docstring")
            .value("Member1", DocstringTestEnum2::Member1)
            .value("Member2", DocstringTestEnum2::Member2);
    }

    {
        py::options options;
        options.disable_enum_members_docstring();

        enum class DocstringTestEnum3 { Member1, Member2 };

        py::enum_<DocstringTestEnum3>(m, "DocstringTestEnum3", "Enum docstring")
            .value("Member1", DocstringTestEnum3::Member1)
            .value("Member2", DocstringTestEnum3::Member2);
    }

    {
        py::options options;
        options.disable_user_defined_docstrings();

        enum class DocstringTestEnum4 { Member1, Member2 };

        py::enum_<DocstringTestEnum4>(m, "DocstringTestEnum4", "Enum docstring")
            .value("Member1", DocstringTestEnum4::Member1)
            .value("Member2", DocstringTestEnum4::Member2);
    }

    {
        py::options options;
        options.disable_user_defined_docstrings();
        options.disable_enum_members_docstring();

        enum class DocstringTestEnum5 { Member1, Member2 };

        py::enum_<DocstringTestEnum5>(m, "DocstringTestEnum5", "Enum docstring")
            .value("Member1", DocstringTestEnum5::Member1)
            .value("Member2", DocstringTestEnum5::Member2);
    }
}
