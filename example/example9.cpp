/*
    example/example9.cpp -- nested modules, importing modules, and
                            internal references

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"

void submodule_func() {
    std::cout << "submodule_func()" << std::endl;
}

class A {
public:
    A(int v) : v(v) { std::cout << "A constructor" << std::endl; }
    ~A() { std::cout << "A destructor" << std::endl; }
    A(const A&) { std::cout << "A copy constructor" << std::endl; }
    std::string toString() { return "A[" + std::to_string(v) + "]"; }
private:
    int v;
};

class B {
public:
    B() { std::cout << "B constructor" << std::endl; }
    ~B() { std::cout << "B destructor" << std::endl; }
    B(const B&) { std::cout << "B copy constructor" << std::endl; }
    A &get_a1() { return a1; }
    A &get_a2() { return a2; }

    A a1{1};
    A a2{2};
};

void init_ex9(py::module &m) {
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
}
