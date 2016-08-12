/*
    tests/test_callbacks.cpp -- callbacks

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include "constructor_stats.h"
#include <pybind11/functional.h>


bool test_callback1(py::object func) {
    func();
    return false;
}

int test_callback2(py::object func) {
    py::object result = func("Hello", 'x', true, 5);
    return result.cast<int>();
}

void test_callback3(const std::function<int(int)> &func) {
    cout << "func(43) = " << func(43)<< std::endl;
}

std::function<int(int)> test_callback4() {
    return [](int i) { return i+1; };
}

py::cpp_function test_callback5() {
    return py::cpp_function([](int i) { return i+1; },
       py::arg("number"));
}

int dummy_function(int i) { return i + 1; }
int dummy_function2(int i, int j) { return i + j; }
std::function<int(int)> roundtrip(std::function<int(int)> f) { 
    if (!f)
        std::cout << "roundtrip (got None).." << std::endl;
    else
        std::cout << "roundtrip.." << std::endl;
    return f;
}

void test_dummy_function(const std::function<int(int)> &f) {
    using fn_type = int (*)(int);
    auto result = f.target<fn_type>();
    if (!result) {
        std::cout << "could not convert to a function pointer." << std::endl;
        auto r = f(1);
        std::cout << "eval(1) = " << r << std::endl;
    } else if (*result == dummy_function) {
        std::cout << "argument matches dummy_function" << std::endl;
        auto r = (*result)(1);
        std::cout << "eval(1) = " << r << std::endl;
    } else {
        std::cout << "argument does NOT match dummy_function. This should never happen!" << std::endl;
    }
}

struct Payload {
    Payload() {
        print_default_created(this);
    }
    ~Payload() {
        print_destroyed(this);
    }
    Payload(const Payload &) {
        print_copy_created(this);
    }
    Payload(Payload &&) {
        print_move_created(this);
    }
};

void init_ex_callbacks(py::module &m) {
    m.def("test_callback1", &test_callback1);
    m.def("test_callback2", &test_callback2);
    m.def("test_callback3", &test_callback3);
    m.def("test_callback4", &test_callback4);
    m.def("test_callback5", &test_callback5);

    /* Test cleanup of lambda closure */

    m.def("test_cleanup", []() -> std::function<void(void)> { 
        Payload p;

        return [p]() {
            /* p should be cleaned up when the returned function is garbage collected */
        };
    });

    /* Test if passing a function pointer from C++ -> Python -> C++ yields the original pointer */
    m.def("dummy_function", &dummy_function);
    m.def("dummy_function2", &dummy_function2);
    m.def("roundtrip", &roundtrip);
    m.def("test_dummy_function", &test_dummy_function);
    // Export the payload constructor statistics for testing purposes:
    m.def("payload_cstats", &ConstructorStats::get<Payload>);
}
