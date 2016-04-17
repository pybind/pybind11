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

void init_ex14(py::module &m) {
    py::class_<py::opaque<StringList>>(m, "StringList")
        .def(py::init<>())
        .def("push_back", [](py::opaque<StringList> &l, const std::string &str) { l->push_back(str); })
        .def("pop_back", [](py::opaque<StringList> &l) { l->pop_back(); })
        .def("back", [](py::opaque<StringList> &l) { return l->back(); });

    m.def("print_opaque_list", [](py::opaque<StringList> &_l) {
        StringList &l = _l;
        std::cout << "Opaque list: " << std::endl;
        for (auto entry : l)
           std::cout << "  " << entry << std::endl; 
    });

    m.def("return_void_ptr", []() { return (void *) 1234; });
    m.def("print_void_ptr", [](void *ptr) { std::cout << "Got void ptr : " << (uint64_t) ptr << std::endl; });
    m.def("return_null_str", []() { return (char *) nullptr; });
    m.def("print_null_str", [](char *ptr) { std::cout << "Got null str : " << (uint64_t) ptr << std::endl; });

    m.def("return_unique_ptr", []() -> std::unique_ptr<StringList> {
        StringList *result = new StringList();
        result->push_back("some value");
        return std::unique_ptr<StringList>(result);
    });
}
