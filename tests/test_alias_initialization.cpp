/*
    tests/test_alias_initialization.cpp -- test cases and example of different trampoline
    initialization modes

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>, Jason Rhinelander <jason@imaginary.ca>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

test_initializer alias_initialization([](py::module &m) {
    // don't invoke Python dispatch classes by default when instantiating C++ classes that were not
    // extended on the Python side
    struct A {
        virtual ~A() {}
        virtual void f() { py::print("A.f()"); }
    };

    struct PyA : A {
        PyA() { py::print("PyA.PyA()"); }
        ~PyA() { py::print("PyA.~PyA()"); }

        void f() override {
            py::print("PyA.f()");
            PYBIND11_OVERLOAD(void, A, f);
        }
    };

    auto call_f = [](A *a) { a->f(); };

    py::class_<A, PyA>(m, "A")
        .def(py::init<>())
        .def("f", &A::f);

    m.def("call_f", call_f);


    // ... unless we explicitly request it, as in this example:
    struct A2 {
        virtual ~A2() {}
        virtual void f() { py::print("A2.f()"); }
    };

    struct PyA2 : A2 {
        PyA2() { py::print("PyA2.PyA2()"); }
        ~PyA2() { py::print("PyA2.~PyA2()"); }
        void f() override {
            py::print("PyA2.f()");
            PYBIND11_OVERLOAD(void, A2, f);
        }
    };

    py::class_<A2, PyA2>(m, "A2")
        .def(py::init_alias<>())
        .def("f", &A2::f);

    m.def("call_f", [](A2 *a2) { a2->f(); });

});

