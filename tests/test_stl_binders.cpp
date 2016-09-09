/*
    tests/test_stl_binders.cpp -- Usage of stl_binders functions

    Copyright (c) 2016 Sergey Lyskov

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

#include <pybind11/stl_bind.h>
#include <map>
#include <unordered_map>

class El {
public:
    El() = delete;
    El(int v) : a(v) { }

    int a;
};

std::ostream & operator<<(std::ostream &s, El const&v) {
    s << "El{" << v.a << '}';
    return s;
}

test_initializer stl_binder_vector([](py::module &m) {
    py::class_<El>(m, "El")
        .def(py::init<int>());

    py::bind_vector<std::vector<unsigned int>>(m, "VectorInt");
    py::bind_vector<std::vector<bool>>(m, "VectorBool");

    py::bind_vector<std::vector<El>>(m, "VectorEl");

    py::bind_vector<std::vector<std::vector<El>>>(m, "VectorVectorEl");
});

test_initializer stl_binder_map([](py::module &m) {
    py::bind_map<std::map<std::string, double>>(m, "MapStringDouble");
    py::bind_map<std::unordered_map<std::string, double>>(m, "UnorderedMapStringDouble");

    py::bind_map<std::map<std::string, double const>>(m, "MapStringDoubleConst");
    py::bind_map<std::unordered_map<std::string, double const>>(m, "UnorderedMapStringDoubleConst");
});
