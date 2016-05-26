/*
    example/example12.cpp -- overriding virtual functions from Python

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"
#include <pybind11/functional.h>

/* This is an example class that we'll want to be able to extend from Python */
class Example12  {
public:
    Example12(int state) : state(state) {
        cout << "Constructing Example12.." << endl;
    }

    ~Example12() {
        cout << "Destructing Example12.." << endl;
    }

    virtual int run(int value) {
        std::cout << "Original implementation of Example12::run(state=" << state
                  << ", value=" << value << ")" << std::endl;
        return state + value;
    }

    virtual bool run_bool() = 0;
    virtual void pure_virtual() = 0;
private:
    int state;
};

/* This is a wrapper class that must be generated */
class PyExample12 : public Example12 {
public:
    using Example12::Example12; /* Inherit constructors */

    virtual int run(int value) {
        /* Generate wrapping code that enables native function overloading */
        PYBIND11_OVERLOAD(
            int,        /* Return type */
            Example12,  /* Parent class */
            run,        /* Name of function */
            value       /* Argument(s) */
        );
    }

    virtual bool run_bool() {
        PYBIND11_OVERLOAD_PURE(
            bool,         /* Return type */
            Example12,    /* Parent class */
            run_bool,     /* Name of function */
                          /* This function has no arguments. The trailing comma
                             in the previous line is needed for some compilers */
        );
    }

    virtual void pure_virtual() {
        PYBIND11_OVERLOAD_PURE(
            void,         /* Return type */
            Example12,    /* Parent class */
            pure_virtual, /* Name of function */
                          /* This function has no arguments. The trailing comma
                             in the previous line is needed for some compilers */
        );
    }
};

int runExample12(Example12 *ex, int value) {
    return ex->run(value);
}

bool runExample12Bool(Example12* ex) {
    return ex->run_bool();
}

void runExample12Virtual(Example12 *ex) {
    ex->pure_virtual();
}

void init_ex12(py::module &m) {
    /* Important: indicate the trampoline class PyExample12 using the third
       argument to py::class_. The second argument with the unique pointer
       is simply the default holder type used by pybind11. */
    py::class_<Example12, std::unique_ptr<Example12>, PyExample12>(m, "Example12")
        .def(py::init<int>())
        /* Reference original class in function definitions */
        .def("run", &Example12::run)
        .def("run_bool", &Example12::run_bool)
        .def("pure_virtual", &Example12::pure_virtual);

    m.def("runExample12", &runExample12);
    m.def("runExample12Bool", &runExample12Bool);
    m.def("runExample12Virtual", &runExample12Virtual);
}
