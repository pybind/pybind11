/*
    example/example11.cpp -- keyword arguments and default values

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"
#include <pybind11/stl.h>

void kw_func(int x, int y) { std::cout << "kw_func(x=" << x << ", y=" << y << ")" << std::endl; }

void kw_func4(const std::vector<int> &entries) {
    std::cout << "kw_func4: ";
    for (int i : entries)
        std::cout << i << " ";
    std::cout << endl;
}

void call_kw_func(py::function f) {
    py::tuple args = py::make_tuple(1234);
    py::dict kwargs;
    kwargs["y"] = py::cast(5678);
    f(*args, **kwargs);
}

void init_ex11(py::module &m) {
    m.def("kw_func", &kw_func, py::arg("x"), py::arg("y"));
    m.def("kw_func2", &kw_func, py::arg("x") = 100, py::arg("y") = 200);
    m.def("kw_func3", [](const char *) { }, py::arg("data") = std::string("Hello world!"));

    /* A fancier default argument */
    std::vector<int> list;
    list.push_back(13);
    list.push_back(17);

    m.def("kw_func4", &kw_func4, py::arg("myList") = list);
    m.def("call_kw_func", &call_kw_func);
}
