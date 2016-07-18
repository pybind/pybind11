/*
    example/example-constants-and-functions.cpp -- global constants and functions, enumerations, raw byte strings

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"

enum EMyEnumeration {
    EFirstEntry = 1,
    ESecondEntry
};

class ExampleWithEnum {
public:
    enum EMode {
        EFirstMode = 1,
        ESecondMode
    };

    static EMode test_function(EMode mode) {
        std::cout << "ExampleWithEnum::test_function(enum=" << mode << ")" << std::endl;
        return mode;
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
    return (float) i / 2.f;
}

py::bytes return_bytes() {
    const char *data = "\x01\x00\x02\x00";
    return std::string(data, 4);
}

void print_bytes(py::bytes bytes) {
    std::string value = (std::string) bytes;
    for (size_t i = 0; i < value.length(); ++i)
        std::cout << "bytes[" << i << "]=" << (int) value[i] << std::endl;
}

void init_ex_constants_and_functions(py::module &m) {
    m.def("test_function", &test_function1);
    m.def("test_function", &test_function2);
    m.def("test_function", &test_function3);
    m.attr("some_constant") = py::int_(14);

    py::enum_<EMyEnumeration>(m, "EMyEnumeration")
        .value("EFirstEntry", EFirstEntry)
        .value("ESecondEntry", ESecondEntry)
        .export_values();

    py::class_<ExampleWithEnum> exenum_class(m, "ExampleWithEnum");
    exenum_class.def_static("test_function", &ExampleWithEnum::test_function);
    py::enum_<ExampleWithEnum::EMode>(exenum_class, "EMode")
        .value("EFirstMode", ExampleWithEnum::EFirstMode)
        .value("ESecondMode", ExampleWithEnum::ESecondMode)
        .export_values();

    m.def("return_bytes", &return_bytes);
    m.def("print_bytes", &print_bytes);
}
