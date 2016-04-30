/*
    example/example14.cpp -- opaque types, passing void pointers

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"
#include <pybind11/stl.h>
#include <vector>

typedef std::vector<std::string> StringList;

class ClassWithSTLVecProperty {
public:
    StringList stringList;
};

/* IMPORTANT: Disable internal pybind11 translation mechanisms for STL data structures */
PYBIND11_MAKE_OPAQUE(StringList);

void init_ex14(py::module &m) {
    py::class_<StringList>(m, "StringList")
        .def(py::init<>())
        .def("pop_back", &StringList::pop_back)
        /* There are multiple versions of push_back(), etc. Select the right ones. */
        .def("push_back", (void (StringList::*)(const std::string &)) &StringList::push_back)
        .def("back", (std::string &(StringList::*)()) &StringList::back)
        .def("__len__", [](const StringList &v) { return v.size(); })
        .def("__iter__", [](StringList &v) {
           return py::make_iterator(v.begin(), v.end());
        }, py::keep_alive<0, 1>());

    py::class_<ClassWithSTLVecProperty>(m, "ClassWithSTLVecProperty")
        .def(py::init<>())
        .def_readwrite("stringList", &ClassWithSTLVecProperty::stringList);

    m.def("print_opaque_list", [](const StringList &l) {
        std::cout << "Opaque list: [";
        bool first = true;
        for (auto entry : l) {
            if (!first)
                std::cout << ", ";
            std::cout << entry;
            first = false;
        }
        std::cout << "]" << std::endl;
    });

    m.def("return_void_ptr", []() { return (void *) 0x1234; });
    m.def("print_void_ptr", [](void *ptr) { std::cout << "Got void ptr : 0x" << std::hex << (uint64_t) ptr << std::endl; });
    m.def("return_null_str", []() { return (char *) nullptr; });
    m.def("print_null_str", [](char *ptr) { std::cout << "Got null str : 0x" << std::hex << (uint64_t) ptr << std::endl; });

    m.def("return_unique_ptr", []() -> std::unique_ptr<StringList> {
        StringList *result = new StringList();
        result->push_back("some value");
        return std::unique_ptr<StringList>(result);
    });
}
