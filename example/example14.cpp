/*
    example/example14.cpp -- opaque types

    Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>

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
}
