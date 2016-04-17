/*
    example/example1.cpp -- constructors, deconstructors, attribute access,
    __str__, argument and return value conventions

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"

class Example1 {
public:
    Example1() {
        cout << "Called Example1 default constructor.." << endl;
    }
    Example1(int value) : value(value) {
        cout << "Called Example1 constructor with value " << value << ".." << endl;
    }
    Example1(const Example1 &e) : value(e.value) {
        cout << "Called Example1 copy constructor with value " << value << ".." << endl;
    }
    Example1(Example1 &&e) : value(e.value) {
        cout << "Called Example1 move constructor with value " << value << ".." << endl;
        e.value = 0;
    }
    ~Example1() {
        cout << "Called Example1 destructor (" << value << ")" << endl;
    }
    std::string toString() {
        return "Example1[value=" + std::to_string(value) + "]";
    }

    void operator=(const Example1 &e) { cout << "Assignment operator" << endl; value = e.value; }
    void operator=(Example1 &&e) { cout << "Move assignment operator" << endl; value = e.value; e.value = 0;}

    void add1(Example1 other) { value += other.value; }           // passing by value
    void add2(Example1 &other) { value += other.value; }          // passing by reference
    void add3(const Example1 &other) { value += other.value; }    // passing by const reference
    void add4(Example1 *other) { value += other->value; }         // passing by pointer
    void add5(const Example1 *other) { value += other->value; }   // passing by const pointer

    void add6(int other) { value += other; }                      // passing by value
    void add7(int &other) { value += other; }                     // passing by reference
    void add8(const int &other) { value += other; }               // passing by const reference
    void add9(int *other) { value += *other; }                    // passing by pointer
    void add10(const int *other) { value += *other; }             // passing by const pointer

    Example1 self1() { return *this; }                            // return by value
    Example1 &self2() { return *this; }                           // return by reference
    const Example1 &self3() { return *this; }                     // return by const reference
    Example1 *self4() { return this; }                            // return by pointer
    const Example1 *self5() { return this; }                      // return by const pointer

    int internal1() { return value; }                             // return by value
    int &internal2() { return value; }                            // return by reference
    const int &internal3() { return value; }                      // return by const reference
    int *internal4() { return &value; }                           // return by pointer
    const int *internal5() { return &value; }                     // return by const pointer

    int value = 0;
};

void init_ex1(py::module &m) {
    py::class_<Example1>(m, "Example1")
        .def(py::init<>())
        .def(py::init<int>())
        .def(py::init<const Example1&>())
        .def("add1", &Example1::add1)
        .def("add2", &Example1::add2)
        .def("add3", &Example1::add3)
        .def("add4", &Example1::add4)
        .def("add5", &Example1::add5)
        .def("add6", &Example1::add6)
        .def("add7", &Example1::add7)
        .def("add8", &Example1::add8)
        .def("add9", &Example1::add9)
        .def("add10", &Example1::add10)
        .def("self1", &Example1::self1)
        .def("self2", &Example1::self2)
        .def("self3", &Example1::self3)
        .def("self4", &Example1::self4)
        .def("self5", &Example1::self5)
        .def("internal1", &Example1::internal1)
        .def("internal2", &Example1::internal2)
        .def("internal3", &Example1::internal3)
        .def("internal4", &Example1::internal4)
        .def("internal5", &Example1::internal5)
        .def("__str__", &Example1::toString)
        .def_readwrite("value", &Example1::value);
}
