/*
    tests/test_kwargs_and_defaults.cpp -- keyword arguments and default values

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include <pybind11/stl.h>

std::string kw_func(int x, int y) { return "x=" + std::to_string(x) + ", y=" + std::to_string(y); }

std::string kw_func4(const std::vector<int> &entries) {
    std::string ret = "{";
    for (int i : entries)
        ret += std::to_string(i) + " ";
    ret.back() = '}';
    return ret;
}

py::tuple args_function(py::args args) {
    return args;
}

py::tuple args_kwargs_function(py::args args, py::kwargs kwargs) {
    return py::make_tuple(args, kwargs);
}

py::tuple mixed_plus_args(int i, double j, py::args args) {
    return py::make_tuple(i, j, args);
}

py::tuple mixed_plus_kwargs(int i, double j, py::kwargs kwargs) {
    return py::make_tuple(i, j, kwargs);
}

py::tuple mixed_plus_args_kwargs(int i, double j, py::args args, py::kwargs kwargs) {
    return py::make_tuple(i, j, args, kwargs);
}

// pybind11 won't allow these to be bound: args and kwargs, if present, must be at the end.
void bad_args1(py::args, int) {}
void bad_args2(py::kwargs, int) {}
void bad_args3(py::kwargs, py::args) {}
void bad_args4(py::args, int, py::kwargs) {}
void bad_args5(py::args, py::kwargs, int) {}
void bad_args6(py::args, py::args) {}
void bad_args7(py::kwargs, py::kwargs) {}

struct KWClass {
    void foo(int, float) {}
};

test_initializer arg_keywords_and_defaults([](py::module &m) {
    m.def("kw_func0", &kw_func);
    m.def("kw_func1", &kw_func, py::arg("x"), py::arg("y"));
    m.def("kw_func2", &kw_func, py::arg("x") = 100, py::arg("y") = 200);
    m.def("kw_func3", [](const char *) { }, py::arg("data") = std::string("Hello world!"));

    /* A fancier default argument */
    std::vector<int> list;
    list.push_back(13);
    list.push_back(17);
    m.def("kw_func4", &kw_func4, py::arg("myList") = list);

    m.def("args_function", &args_function);
    m.def("args_kwargs_function", &args_kwargs_function);

    m.def("kw_func_udl", &kw_func, "x"_a, "y"_a=300);
    m.def("kw_func_udl_z", &kw_func, "x"_a, "y"_a=0);

    py::class_<KWClass>(m, "KWClass")
        .def("foo0", &KWClass::foo)
        .def("foo1", &KWClass::foo, "x"_a, "y"_a);

    m.def("mixed_plus_args", &mixed_plus_args);
    m.def("mixed_plus_kwargs", &mixed_plus_kwargs);
    m.def("mixed_plus_args_kwargs", &mixed_plus_args_kwargs);

    m.def("mixed_plus_args_kwargs_defaults", &mixed_plus_args_kwargs,
            py::arg("i") = 1, py::arg("j") = 3.14159);

    // Uncomment these to test that the static_assert is indeed working:
//    m.def("bad_args1", &bad_args1);
//    m.def("bad_args2", &bad_args2);
//    m.def("bad_args3", &bad_args3);
//    m.def("bad_args4", &bad_args4);
//    m.def("bad_args5", &bad_args5);
//    m.def("bad_args6", &bad_args6);
//    m.def("bad_args7", &bad_args7);
});
