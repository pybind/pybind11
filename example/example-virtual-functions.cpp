/*
    example/example-virtual-functions.cpp -- overriding virtual functions from Python

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"
#include <pybind11/functional.h>

/* This is an example class that we'll want to be able to extend from Python */
class ExampleVirt  {
public:
    ExampleVirt(int state) : state(state) {
        cout << "Constructing ExampleVirt.." << endl;
    }

    ~ExampleVirt() {
        cout << "Destructing ExampleVirt.." << endl;
    }

    virtual int run(int value) {
        std::cout << "Original implementation of ExampleVirt::run(state=" << state
                  << ", value=" << value << ")" << std::endl;
        return state + value;
    }

    virtual bool run_bool() = 0;
    virtual void pure_virtual() = 0;
private:
    int state;
};

/* This is a wrapper class that must be generated */
class PyExampleVirt : public ExampleVirt {
public:
    using ExampleVirt::ExampleVirt; /* Inherit constructors */

    virtual int run(int value) {
        /* Generate wrapping code that enables native function overloading */
        PYBIND11_OVERLOAD(
            int,         /* Return type */
            ExampleVirt, /* Parent class */
            run,         /* Name of function */
            value        /* Argument(s) */
        );
    }

    virtual bool run_bool() {
        PYBIND11_OVERLOAD_PURE(
            bool,         /* Return type */
            ExampleVirt,  /* Parent class */
            run_bool,     /* Name of function */
                          /* This function has no arguments. The trailing comma
                             in the previous line is needed for some compilers */
        );
    }

    virtual void pure_virtual() {
        PYBIND11_OVERLOAD_PURE(
            void,         /* Return type */
            ExampleVirt,  /* Parent class */
            pure_virtual, /* Name of function */
                          /* This function has no arguments. The trailing comma
                             in the previous line is needed for some compilers */
        );
    }
};

int runExampleVirt(ExampleVirt *ex, int value) {
    return ex->run(value);
}

bool runExampleVirtBool(ExampleVirt* ex) {
    return ex->run_bool();
}

void runExampleVirtVirtual(ExampleVirt *ex) {
    ex->pure_virtual();
}

void init_ex_virtual_functions(py::module &m) {
    /* Important: indicate the trampoline class PyExampleVirt using the third
       argument to py::class_. The second argument with the unique pointer
       is simply the default holder type used by pybind11. */
    py::class_<ExampleVirt, std::unique_ptr<ExampleVirt>, PyExampleVirt>(m, "ExampleVirt")
        .def(py::init<int>())
        /* Reference original class in function definitions */
        .def("run", &ExampleVirt::run)
        .def("run_bool", &ExampleVirt::run_bool)
        .def("pure_virtual", &ExampleVirt::pure_virtual);

    m.def("runExampleVirt", &runExampleVirt);
    m.def("runExampleVirtBool", &runExampleVirtBool);
    m.def("runExampleVirtVirtual", &runExampleVirtVirtual);
}
