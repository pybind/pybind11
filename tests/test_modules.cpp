/*
    tests/test_modules.cpp -- nested modules, importing modules, and
                            internal references

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include "constructor_stats.h"

std::string submodule_func() {
    return "submodule_func()";
}

class A {
public:
    A(int v) : v(v) { print_created(this, v); }
    ~A() { print_destroyed(this); }
    A(const A&) { print_copy_created(this); }
    A& operator=(const A &copy) { print_copy_assigned(this); v = copy.v; return *this; }
    std::string toString() { return "A[" + std::to_string(v) + "]"; }
private:
    int v;
};

class B {
public:
    B() { print_default_created(this); }
    ~B() { print_destroyed(this); }
    B(const B&) { print_copy_created(this); }
    B& operator=(const B &copy) { print_copy_assigned(this); a1 = copy.a1; a2 = copy.a2; return *this; }
    A &get_a1() { return a1; }
    A &get_a2() { return a2; }

    A a1{1};
    A a2{2};
};

test_initializer modules([](py::module &m) {
    py::module m_sub = m.def_submodule("submodule");
    m_sub.def("submodule_func", &submodule_func);

    py::class_<A>(m_sub, "A")
        .def(py::init<int>())
        .def("__repr__", &A::toString);

    py::class_<B>(m_sub, "B")
        .def(py::init<>())
        .def("get_a1", &B::get_a1, "Return the internal A 1", py::return_value_policy::reference_internal)
        .def("get_a2", &B::get_a2, "Return the internal A 2", py::return_value_policy::reference_internal)
        .def_readwrite("a1", &B::a1)  // def_readonly uses an internal reference return policy by default
        .def_readwrite("a2", &B::a2);

    m.attr("OD") = py::module::import("collections").attr("OrderedDict");
});
