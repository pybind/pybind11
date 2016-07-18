/*
    example/example-methods-and-attributes.cpp -- constructors, deconstructors, attribute access,
    __str__, argument and return value conventions

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"

class ExampleMandA {
public:
    ExampleMandA() {
        cout << "Called ExampleMandA default constructor.." << endl;
    }
    ExampleMandA(int value) : value(value) {
        cout << "Called ExampleMandA constructor with value " << value << ".." << endl;
    }
    ExampleMandA(const ExampleMandA &e) : value(e.value) {
        cout << "Called ExampleMandA copy constructor with value " << value << ".." << endl;
    }
    ExampleMandA(ExampleMandA &&e) : value(e.value) {
        cout << "Called ExampleMandA move constructor with value " << value << ".." << endl;
        e.value = 0;
    }
    ~ExampleMandA() {
        cout << "Called ExampleMandA destructor (" << value << ")" << endl;
    }
    std::string toString() {
        return "ExampleMandA[value=" + std::to_string(value) + "]";
    }

    void operator=(const ExampleMandA &e) { cout << "Assignment operator" << endl; value = e.value; }
    void operator=(ExampleMandA &&e) { cout << "Move assignment operator" << endl; value = e.value; e.value = 0;}

    void add1(ExampleMandA other) { value += other.value; }           // passing by value
    void add2(ExampleMandA &other) { value += other.value; }          // passing by reference
    void add3(const ExampleMandA &other) { value += other.value; }    // passing by const reference
    void add4(ExampleMandA *other) { value += other->value; }         // passing by pointer
    void add5(const ExampleMandA *other) { value += other->value; }   // passing by const pointer

    void add6(int other) { value += other; }                      // passing by value
    void add7(int &other) { value += other; }                     // passing by reference
    void add8(const int &other) { value += other; }               // passing by const reference
    void add9(int *other) { value += *other; }                    // passing by pointer
    void add10(const int *other) { value += *other; }             // passing by const pointer

    ExampleMandA self1() { return *this; }                            // return by value
    ExampleMandA &self2() { return *this; }                           // return by reference
    const ExampleMandA &self3() { return *this; }                     // return by const reference
    ExampleMandA *self4() { return this; }                            // return by pointer
    const ExampleMandA *self5() { return this; }                      // return by const pointer

    int internal1() { return value; }                             // return by value
    int &internal2() { return value; }                            // return by reference
    const int &internal3() { return value; }                      // return by const reference
    int *internal4() { return &value; }                           // return by pointer
    const int *internal5() { return &value; }                     // return by const pointer

    int value = 0;
};

void init_ex_methods_and_attributes(py::module &m) {
    py::class_<ExampleMandA>(m, "ExampleMandA")
        .def(py::init<>())
        .def(py::init<int>())
        .def(py::init<const ExampleMandA&>())
        .def("add1", &ExampleMandA::add1)
        .def("add2", &ExampleMandA::add2)
        .def("add3", &ExampleMandA::add3)
        .def("add4", &ExampleMandA::add4)
        .def("add5", &ExampleMandA::add5)
        .def("add6", &ExampleMandA::add6)
        .def("add7", &ExampleMandA::add7)
        .def("add8", &ExampleMandA::add8)
        .def("add9", &ExampleMandA::add9)
        .def("add10", &ExampleMandA::add10)
        .def("self1", &ExampleMandA::self1)
        .def("self2", &ExampleMandA::self2)
        .def("self3", &ExampleMandA::self3)
        .def("self4", &ExampleMandA::self4)
        .def("self5", &ExampleMandA::self5)
        .def("internal1", &ExampleMandA::internal1)
        .def("internal2", &ExampleMandA::internal2)
        .def("internal3", &ExampleMandA::internal3)
        .def("internal4", &ExampleMandA::internal4)
        .def("internal5", &ExampleMandA::internal5)
        .def("__str__", &ExampleMandA::toString)
        .def_readwrite("value", &ExampleMandA::value);
}
