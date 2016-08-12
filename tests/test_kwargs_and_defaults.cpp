/*
    tests/test_kwargs_and_defaults.cpp -- keyword arguments and default values

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include <pybind11/stl.h>

void kw_func(int x, int y) { std::cout << "kw_func(x=" << x << ", y=" << y << ")" << std::endl; }

void kw_func4(const std::vector<int> &entries) {
    std::cout << "kw_func4: ";
    for (int i : entries)
        std::cout << i << " ";
    std::cout << endl;
}

py::object call_kw_func(py::function f) {
    py::tuple args = py::make_tuple(1234);
    py::dict kwargs;
    kwargs["y"] = py::cast(5678);
    return f(*args, **kwargs);
}

void args_function(py::args args) {
    for (size_t it=0; it<args.size(); ++it)
        std::cout << "got argument: " << py::object(args[it]) << std::endl;
}

void args_kwargs_function(py::args args, py::kwargs kwargs) {
    for (auto item : args)
        std::cout << "got argument: " << item << std::endl;
    if (kwargs) {
        for (auto item : kwargs)
            std::cout << "got keyword argument: " << item.first << " -> " << item.second << std::endl;
    }
}

struct KWClass {
    void foo(int, float) {}
};

void init_ex_arg_keywords_and_defaults(py::module &m) {
    m.def("kw_func0", &kw_func);
    m.def("kw_func1", &kw_func, py::arg("x"), py::arg("y"));
    m.def("kw_func2", &kw_func, py::arg("x") = 100, py::arg("y") = 200);
    m.def("kw_func3", [](const char *) { }, py::arg("data") = std::string("Hello world!"));

    /* A fancier default argument */
    std::vector<int> list;
    list.push_back(13);
    list.push_back(17);

    m.def("kw_func4", &kw_func4, py::arg("myList") = list);
    m.def("call_kw_func", &call_kw_func);

    m.def("args_function", &args_function);
    m.def("args_kwargs_function", &args_kwargs_function);

    using namespace py::literals;
    m.def("kw_func_udl", &kw_func, "x"_a, "y"_a=300);
    m.def("kw_func_udl_z", &kw_func, "x"_a, "y"_a=0);
    
    py::class_<KWClass>(m, "KWClass")
        .def("foo0", &KWClass::foo)
        .def("foo1", &KWClass::foo, "x"_a, "y"_a);
}
