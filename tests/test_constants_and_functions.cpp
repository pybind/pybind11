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

// Test that we properly handle C++17 exception specifiers (which are part of the function signature
// in C++17).  These should all still work before C++17, but don't affect the function signature.
namespace test_exc_sp {
int f1(int x) noexcept { return x+1; }
int f2(int x) noexcept(true) { return x+2; }
int f3(int x) noexcept(false) { return x+3; }
int f4(int x) throw() { return x+4; } // Deprecated equivalent to noexcept(true)
struct C {
    int m1(int x) noexcept { return x-1; }
    int m2(int x) const noexcept { return x-2; }
    int m3(int x) noexcept(true) { return x-3; }
    int m4(int x) const noexcept(true) { return x-4; }
    int m5(int x) noexcept(false) { return x-5; }
    int m6(int x) const noexcept(false) { return x-6; }
    int m7(int x) throw() { return x-7; }
    int m8(int x) const throw() { return x-8; }
};
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

    using namespace test_exc_sp;
    py::module m2 = m.def_submodule("exc_sp");
    py::class_<C>(m2, "C")
        .def(py::init<>())
        .def("m1", &C::m1)
        .def("m2", &C::m2)
        .def("m3", &C::m3)
        .def("m4", &C::m4)
        .def("m5", &C::m5)
        .def("m6", &C::m6)
        .def("m7", &C::m7)
        .def("m8", &C::m8)
        ;
    m2.def("f1", f1);
    m2.def("f2", f2);
    m2.def("f3", f3);
    m2.def("f4", f4);
});
