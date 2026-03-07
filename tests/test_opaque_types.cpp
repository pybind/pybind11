/*
    tests/test_opaque_types.cpp -- opaque types, passing void pointers

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <pybind11/stl.h>

#include "pybind11_tests.h"

#include <array>
#include <vector>

// IMPORTANT: Disable internal pybind11 translation mechanisms for STL data structures
//
// This also deliberately doesn't use the below StringList type alias to test
// that MAKE_OPAQUE can handle a type containing a `,`.  (The `std::allocator`
// bit is just the default `std::vector` allocator).
PYBIND11_MAKE_OPAQUE(std::vector<std::string, std::allocator<std::string>>)

// Test for GitHub issue #5988: PYBIND11_MAKE_OPAQUE with std::array types.
// These types are not used as converted types in other test files, so they
// can safely be made opaque here without ODR violations.
PYBIND11_MAKE_OPAQUE(std::array<double, 3>)
PYBIND11_MAKE_OPAQUE(std::vector<std::array<double, 3>>)

using StringList = std::vector<std::string, std::allocator<std::string>>;

// Type aliases for issue #5988 test
using Array3d = std::array<double, 3>;
using VecArray3d = std::vector<Array3d>;

TEST_SUBMODULE(opaque_types, m) {
    // test_string_list
    py::class_<StringList>(m, "StringList")
        .def(py::init<>())
        .def("pop_back", &StringList::pop_back)
        /* There are multiple versions of push_back(), etc. Select the right ones. */
        .def("push_back", (void (StringList::*)(const std::string &)) &StringList::push_back)
        .def("back", (std::string & (StringList::*) ()) & StringList::back)
        .def("__len__", [](const StringList &v) { return v.size(); })
        .def(
            "__iter__",
            [](StringList &v) { return py::make_iterator(v.begin(), v.end()); },
            py::keep_alive<0, 1>());

    class ClassWithSTLVecProperty {
    public:
        StringList stringList;
    };
    py::class_<ClassWithSTLVecProperty>(m, "ClassWithSTLVecProperty")
        .def(py::init<>())
        .def_readwrite("stringList", &ClassWithSTLVecProperty::stringList);

    m.def("print_opaque_list", [](const StringList &l) {
        std::string ret = "Opaque list: [";
        bool first = true;
        for (const auto &entry : l) {
            if (!first) {
                ret += ", ";
            }
            ret += entry;
            first = false;
        }
        return ret + "]";
    });

    // test_pointers
    m.def("return_void_ptr", []() { return (void *) 0x1234; });
    m.def("get_void_ptr_value", [](void *ptr) { return reinterpret_cast<std::intptr_t>(ptr); });
    m.def("return_null_str", []() { return (char *) nullptr; });
    m.def("get_null_str_value", [](char *ptr) { return reinterpret_cast<std::intptr_t>(ptr); });

    m.def("return_unique_ptr", []() -> std::unique_ptr<StringList> {
        auto *result = new StringList();
        result->emplace_back("some value");
        return std::unique_ptr<StringList>(result);
    });

    // test unions
    py::class_<IntFloat>(m, "IntFloat")
        .def(py::init<>())
        .def_readwrite("i", &IntFloat::i)
        .def_readwrite("f", &IntFloat::f);

    // test_issue_5988: PYBIND11_MAKE_OPAQUE with std::array and nested containers
    // (Regression test for crash when importing modules with opaque std::array types)
    py::class_<Array3d>(m, "Array3d")
        .def(py::init<>())
        .def("__getitem__",
             [](const Array3d &a, std::size_t i) -> double {
                 if (i >= a.size()) {
                     throw py::index_error();
                 }
                 return a[i];
             })
        .def("__setitem__",
             [](Array3d &a, std::size_t i, double v) {
                 if (i >= a.size()) {
                     throw py::index_error();
                 }
                 a[i] = v;
             })
        .def("__len__", [](const Array3d &a) { return a.size(); });

    py::class_<VecArray3d>(m, "VecArray3d")
        .def(py::init<>())
        .def("push_back", [](VecArray3d &v, const Array3d &a) { v.push_back(a); })
        .def("__getitem__",
             [](const VecArray3d &v, std::size_t i) -> const Array3d & {
                 if (i >= v.size()) {
                     throw py::index_error();
                 }
                 return v[i];
             },
             py::return_value_policy::reference_internal)
        .def("__len__", [](const VecArray3d &v) { return v.size(); });
}
