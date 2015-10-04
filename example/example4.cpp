/*
    example/example4.cpp -- global constants and functions, enumerations

    Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"

enum EMyEnumeration {
    EFirstEntry = 1,
    ESecondEntry
};

class Example4 {
public:
    enum EMode {
        EFirstMode = 1,
        ESecondMode
    };

    static void test_function(EMode mode) {
        std::cout << "Example4::test_function(enum=" << mode << ")" << std::endl;
    }
};

bool test_function1() {
    std::cout << "test_function()" << std::endl;
    return false;
}

void test_function2(EMyEnumeration k) {
    std::cout << "test_function(enum=" << k << ")" << std::endl;
}

float test_function3(int i) {
    std::cout << "test_function(" << i << ")" << std::endl;
    return i / 2.f;
}

void init_ex4(py::module &m) {
    m.def("test_function", &test_function1);
    m.def("test_function", &test_function2);
    m.def("test_function", &test_function3);
    m.attr("some_constant") = py::int_(14);

    py::enum_<EMyEnumeration>(m, "EMyEnumeration")
        .value("EFirstEntry", EFirstEntry)
        .value("ESecondEntry", ESecondEntry)
        .export_values();

    py::class_<Example4> ex4_class(m, "Example4");
    ex4_class.def_static("test_function", &Example4::test_function);
    py::enum_<Example4::EMode>(ex4_class, "EMode")
        .value("EFirstMode", Example4::EFirstMode)
        .value("ESecondMode", Example4::ESecondMode)
        .export_values();
}
