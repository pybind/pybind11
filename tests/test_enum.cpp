/*
    tests/test_enums.cpp -- enumerations

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

enum UnscopedEnum {
    EOne = 1,
    ETwo
};

enum class ScopedEnum {
    Two = 2,
    Three
};

enum Flags {
    Read = 4,
    Write = 2,
    Execute = 1
};

class ClassWithUnscopedEnum {
public:
    enum EMode {
        EFirstMode = 1,
        ESecondMode
    };

    static EMode test_function(EMode mode) {
        return mode;
    }
};

std::string test_scoped_enum(ScopedEnum z) {
    return "ScopedEnum::" + std::string(z == ScopedEnum::Two ? "Two" : "Three");
}

test_initializer enums([](py::module &m) {
    m.def("test_scoped_enum", &test_scoped_enum);

    auto e = py::enum_<UnscopedEnum>(m, "UnscopedEnum", py::arithmetic())
        .value("EOne", EOne)
        .value("ETwo", ETwo)
        .export_values()
        .into_class()
        .def("x", [](const UnscopedEnum& e) { return static_cast<int>(e) + 1; })
        .def_property_readonly("y", [](const UnscopedEnum& e) { return static_cast<int>(e) + 2; })
        .def_static("a", []() { return 41; })
        .def_property_readonly_static("b", [](py::object /* unused */) { return 42; });

    auto scoped_enum = py::enum_<ScopedEnum>(m, "ScopedEnum", py::arithmetic());
    scoped_enum
        .value("Two", ScopedEnum::Two)
        .value("Three", ScopedEnum::Three);

    py::setattr(e, "Alias", scoped_enum);

    py::enum_<Flags>(m, "Flags", py::arithmetic())
        .value("Read", Flags::Read)
        .value("Write", Flags::Write)
        .value("Execute", Flags::Execute)
        .export_values();

    py::class_<ClassWithUnscopedEnum> exenum_class(m, "ClassWithUnscopedEnum");
    exenum_class.def_static("test_function", &ClassWithUnscopedEnum::test_function);
    py::enum_<ClassWithUnscopedEnum::EMode>(exenum_class, "EMode")
        .value("EFirstMode", ClassWithUnscopedEnum::EFirstMode)
        .value("ESecondMode", ClassWithUnscopedEnum::ESecondMode)
        .export_values();

    m.def("test_enum_to_int", [](int) { });
    m.def("test_enum_to_uint", [](uint32_t) { });
    m.def("test_enum_to_long_long", [](long long) { });
});
