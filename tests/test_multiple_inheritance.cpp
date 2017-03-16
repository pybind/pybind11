/*
    tests/test_multiple_inheritance.cpp -- multiple inheritance,
    implicit MI casts

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

struct Base1 {
    Base1(int i) : i(i) { }
    int foo() { return i; }
    int i;
};

struct Base2 {
    Base2(int i) : i(i) { }
    int bar() { return i; }
    int i;
};

struct Base12 : Base1, Base2 {
    Base12(int i, int j) : Base1(i), Base2(j) { }
};

struct MIType : Base12 {
    MIType(int i, int j) : Base12(i, j) { }
};

test_initializer multiple_inheritance([](py::module &m) {
    py::class_<Base1> b1(m, "Base1");
    b1.def(py::init<int>())
      .def("foo", &Base1::foo);

    py::class_<Base2> b2(m, "Base2");
    b2.def(py::init<int>())
      .def("bar", &Base2::bar);

    py::class_<Base12, Base1, Base2>(m, "Base12");

    py::class_<MIType, Base12>(m, "MIType")
        .def(py::init<int, int>());

    // Uncommenting this should result in a compile time failure (MI can only be specified via
    // template parameters because pybind has to know the types involved; see discussion in #742 for
    // details).
//    struct Base12v2 : Base1, Base2 {
//        Base12v2(int i, int j) : Base1(i), Base2(j) { }
//    };
//    py::class_<Base12v2>(m, "Base12v2", b1, b2)
//        .def(py::init<int, int>());
});

/* Test the case where not all base classes are specified,
   and where pybind11 requires the py::multiple_inheritance
   flag to perform proper casting between types */

struct Base1a {
    Base1a(int i) : i(i) { }
    int foo() { return i; }
    int i;
};

struct Base2a {
    Base2a(int i) : i(i) { }
    int bar() { return i; }
    int i;
};

struct Base12a : Base1a, Base2a {
    Base12a(int i, int j) : Base1a(i), Base2a(j) { }
};

test_initializer multiple_inheritance_nonexplicit([](py::module &m) {
    py::class_<Base1a, std::shared_ptr<Base1a>>(m, "Base1a")
        .def(py::init<int>())
        .def("foo", &Base1a::foo);

    py::class_<Base2a, std::shared_ptr<Base2a>>(m, "Base2a")
        .def(py::init<int>())
        .def("bar", &Base2a::bar);

    py::class_<Base12a, /* Base1 missing */ Base2a,
               std::shared_ptr<Base12a>>(m, "Base12a", py::multiple_inheritance())
        .def(py::init<int, int>());

    m.def("bar_base2a", [](Base2a *b) { return b->bar(); });
    m.def("bar_base2a_sharedptr", [](std::shared_ptr<Base2a> b) { return b->bar(); });
});

struct Vanilla {
    std::string vanilla() { return "Vanilla"; };
};

struct WithStatic1 {
    static std::string static_func1() { return "WithStatic1"; };
    static int static_value1;
};

struct WithStatic2 {
    static std::string static_func2() { return "WithStatic2"; };
    static int static_value2;
};

struct WithDict { };

struct VanillaStaticMix1 : Vanilla, WithStatic1, WithStatic2 {
    static std::string static_func() { return "VanillaStaticMix1"; }
    static int static_value;
};

struct VanillaStaticMix2 : WithStatic1, Vanilla, WithStatic2 {
    static std::string static_func() { return "VanillaStaticMix2"; }
    static int static_value;
};

struct VanillaDictMix1 : Vanilla, WithDict { };
struct VanillaDictMix2 : WithDict, Vanilla { };

int WithStatic1::static_value1 = 1;
int WithStatic2::static_value2 = 2;
int VanillaStaticMix1::static_value = 12;
int VanillaStaticMix2::static_value = 12;

test_initializer mi_static_properties([](py::module &pm) {
    auto m = pm.def_submodule("mi");

    py::class_<Vanilla>(m, "Vanilla")
        .def(py::init<>())
        .def("vanilla", &Vanilla::vanilla);

    py::class_<WithStatic1>(m, "WithStatic1")
        .def(py::init<>())
        .def_static("static_func1", &WithStatic1::static_func1)
        .def_readwrite_static("static_value1", &WithStatic1::static_value1);

    py::class_<WithStatic2>(m, "WithStatic2")
        .def(py::init<>())
        .def_static("static_func2", &WithStatic2::static_func2)
        .def_readwrite_static("static_value2", &WithStatic2::static_value2);

    py::class_<VanillaStaticMix1, Vanilla, WithStatic1, WithStatic2>(
        m, "VanillaStaticMix1")
        .def(py::init<>())
        .def_static("static_func", &VanillaStaticMix1::static_func)
        .def_readwrite_static("static_value", &VanillaStaticMix1::static_value);

    py::class_<VanillaStaticMix2, WithStatic1, Vanilla, WithStatic2>(
        m, "VanillaStaticMix2")
        .def(py::init<>())
        .def_static("static_func", &VanillaStaticMix2::static_func)
        .def_readwrite_static("static_value", &VanillaStaticMix2::static_value);

#if !defined(PYPY_VERSION)
    py::class_<WithDict>(m, "WithDict", py::dynamic_attr()).def(py::init<>());
    py::class_<VanillaDictMix1, Vanilla, WithDict>(m, "VanillaDictMix1").def(py::init<>());
    py::class_<VanillaDictMix2, WithDict, Vanilla>(m, "VanillaDictMix2").def(py::init<>());
#endif
});
