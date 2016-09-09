/*
    tests/test_keep_alive.cpp -- keep_alive modifier (pybind11's version
    of Boost.Python's with_custodian_and_ward / with_custodian_and_ward_postcall)

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

class Child {
public:
    Child() { py::print("Allocating child."); }
    ~Child() { py::print("Releasing child."); }
};

class Parent {
public:
    Parent() { py::print("Allocating parent."); }
    ~Parent() { py::print("Releasing parent."); }
    void addChild(Child *) { }
    Child *returnChild() { return new Child(); }
    Child *returnNullChild() { return nullptr; }
};

test_initializer keep_alive([](py::module &m) {
    py::class_<Parent>(m, "Parent")
        .def(py::init<>())
        .def("addChild", &Parent::addChild)
        .def("addChildKeepAlive", &Parent::addChild, py::keep_alive<1, 2>())
        .def("returnChild", &Parent::returnChild)
        .def("returnChildKeepAlive", &Parent::returnChild, py::keep_alive<1, 0>())
        .def("returnNullChildKeepAliveChild", &Parent::returnNullChild, py::keep_alive<1, 0>())
        .def("returnNullChildKeepAliveParent", &Parent::returnNullChild, py::keep_alive<0, 1>());

    py::class_<Child>(m, "Child")
        .def(py::init<>());
});
