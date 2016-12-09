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
    py::class_<Base1>(m, "Base1")
        .def(py::init<int>())
        .def("foo", &Base1::foo);

    py::class_<Base2>(m, "Base2")
        .def(py::init<int>())
        .def("bar", &Base2::bar);

    py::class_<Base12, Base1, Base2>(m, "Base12");

    py::class_<MIType, Base12>(m, "MIType")
        .def(py::init<int, int>());
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
