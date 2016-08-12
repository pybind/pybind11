/*
    tests/test_constants_and_functions.cpp -- global constants and functions, enumerations, raw byte strings

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

enum EMyEnumeration {
    EFirstEntry = 1,
    ESecondEntry
};

enum class ECMyEnum {
    Two = 2,
    Three
};

class ExampleWithEnum {
public:
    enum EMode {
        EFirstMode = 1,
        ESecondMode
    };

    static EMode test_function(EMode mode) {
        return mode;
    }
};

std::string test_function1() {
    return "test_function()";
}

std::string test_function2(EMyEnumeration k) {
    return "test_function(enum=" + std::to_string(k) + ")";
}

std::string test_function3(int i) {
    return "test_function(" + std::to_string(i) + ")";
}

std::string test_ecenum(ECMyEnum z) {
    return "test_ecenum(ECMyEnum::" + std::string(z == ECMyEnum::Two ? "Two" : "Three") + ")";
}

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

void init_ex_constants_and_functions(py::module &m) {
    m.def("test_function", &test_function1);
    m.def("test_function", &test_function2);
    m.def("test_function", &test_function3);
    m.def("test_ecenum", &test_ecenum);
    m.attr("some_constant") = py::int_(14);

    py::enum_<EMyEnumeration>(m, "EMyEnumeration")
        .value("EFirstEntry", EFirstEntry)
        .value("ESecondEntry", ESecondEntry)
        .export_values();

    py::enum_<ECMyEnum>(m, "ECMyEnum")
        .value("Two", ECMyEnum::Two)
        .value("Three", ECMyEnum::Three)
        ;

    py::class_<ExampleWithEnum> exenum_class(m, "ExampleWithEnum");
    exenum_class.def_static("test_function", &ExampleWithEnum::test_function);
    py::enum_<ExampleWithEnum::EMode>(exenum_class, "EMode")
        .value("EFirstMode", ExampleWithEnum::EFirstMode)
        .value("ESecondMode", ExampleWithEnum::ESecondMode)
        .export_values();

    m.def("return_bytes", &return_bytes);
    m.def("print_bytes", &print_bytes);
}
