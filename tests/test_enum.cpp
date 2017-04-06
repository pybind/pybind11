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

enum Py3Enum {
    A = -42,
    B = 1,
    C = 42,
};

enum class Py3EnumScoped : short {
    X = 10,
    Y = -1024,
};

enum class Py3EnumEmpty {};

enum class Py3EnumNonUnique {
    X = 1
};

class DummyScope {};

std::string test_scoped_enum(ScopedEnum z) {
    return "ScopedEnum::" + std::string(z == ScopedEnum::Two ? "Two" : "Three");
}

test_initializer enums([](py::module &m) {
    m.def("test_scoped_enum", &test_scoped_enum);

    py::enum_<UnscopedEnum>(m, "UnscopedEnum", py::arithmetic())
        .value("EOne", EOne)
        .value("ETwo", ETwo)
        .export_values();

    py::enum_<ScopedEnum>(m, "ScopedEnum", py::arithmetic())
        .value("Two", ScopedEnum::Two)
        .value("Three", ScopedEnum::Three);

    py::enum_<Flags>(m, "Flags", py::arithmetic())
        .value("Read", Flags::Read)
        .value("Write", Flags::Write)
        .value("Execute", Flags::Execute)
        .export_values();

#if PY_VERSION_HEX >= 0x03000000
    auto scope = py::class_<DummyScope>(m, "DummyScope");
    py::py3_enum<Py3EnumEmpty>(scope, "Py3EnumEmpty");

    auto e = py::py3_enum<Py3Enum>(m, "Py3Enum")
        .value("A", Py3Enum::A)
        .value("B", Py3Enum::B)
        .value("C", Py3Enum::C)
        .extend()
        .def("add", [](Py3Enum x, int y) { return static_cast<int>(x) + y; })
        .def_property_readonly("is_b", [](Py3Enum e) { return e == Py3Enum::B; })
        .def_property_readonly_static("ultimate_answer", [](py::object) { return 42; });

    py::py3_enum<Py3EnumScoped>(m, "Py3EnumScoped")
        .value("X", Py3EnumScoped::X)
        .value("Y", Py3EnumScoped::Y);

    m.def("make_py3_enum", [](bool x) {
        return x ? Py3EnumScoped::X : Py3EnumScoped::Y;
    });

    m.def("take_py3_enum", [](Py3EnumScoped x) {
        return x == Py3EnumScoped::X;
    });

    m.def("non_unique_py3_enum", [=]() {
        py::py3_enum<Py3EnumNonUnique>(m, "Py3EnumNonUnique")
            .value("X", Py3EnumNonUnique::X)
            .value("Y", Py3EnumNonUnique::X);
    });
#endif

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
