/*
    tests/test_constants_and_functions.cpp -- global constants and functions, enumerations, raw byte strings

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

enum MyEnum { EFirstEntry = 1, ESecondEntry };

std::string test_function1() {
    return "test_function()";
}

std::string test_function2(MyEnum k) {
    return "test_function(enum=" + std::to_string(k) + ")";
}

std::string test_function3(int i) {
    return "test_function(" + std::to_string(i) + ")";
}

py::str test_function4(int, float) { return "test_function(int, float)"; }
py::str test_function4(float, int) { return "test_function(float, int)"; }

py::bytes return_bytes() {
    const char *data = "\x01\x00\x02\x00";
    return std::string(data, 4);
}

std::string print_bytes(py::bytes bytes) {
    std::string ret = "bytes[";
    const auto value = static_cast<std::string>(bytes);
    for (size_t i = 0; i < value.length(); ++i) {
        ret += std::to_string(static_cast<int>(value[i])) + " ";
    }
    ret.back() = ']';
    return ret;
}

test_initializer constants_and_functions([](py::module &m) {
    m.attr("some_constant") = py::int_(14);

    m.def("test_function", &test_function1);
    m.def("test_function", &test_function2);
    m.def("test_function", &test_function3);

#if defined(PYBIND11_OVERLOAD_CAST)
    m.def("test_function", py::overload_cast<int, float>(&test_function4));
    m.def("test_function", py::overload_cast<float, int>(&test_function4));
#else
    m.def("test_function", static_cast<py::str (*)(int, float)>(&test_function4));
    m.def("test_function", static_cast<py::str (*)(float, int)>(&test_function4));
#endif

    py::enum_<MyEnum>(m, "MyEnum")
        .value("EFirstEntry", EFirstEntry)
        .value("ESecondEntry", ESecondEntry)
        .export_values();

    m.def("return_bytes", &return_bytes);
    m.def("print_bytes", &print_bytes);
});
