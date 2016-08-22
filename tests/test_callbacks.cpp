/*
    tests/test_callbacks.cpp -- callbacks

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include "constructor_stats.h"
#include <pybind11/functional.h>


py::object test_callback1(py::object func) {
    return func();
}

py::tuple test_callback2(py::object func) {
    return func("Hello", 'x', true, 5);
}

std::string test_callback3(const std::function<int(int)> &func) {
    return "func(43) = " + std::to_string(func(43));
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
std::function<int(int)> roundtrip(std::function<int(int)> f, bool expect_none = false) {
    if (expect_none && f) {
        throw std::runtime_error("Expected None to be converted to empty std::function");
    }
    return f;
}

std::string test_dummy_function(const std::function<int(int)> &f) {
    using fn_type = int (*)(int);
    auto result = f.target<fn_type>();
    if (!result) {
        auto r = f(1);
        return "can't convert to function pointer: eval(1) = " + std::to_string(r);
    } else if (*result == dummy_function) {
        auto r = (*result)(1);
        return "matches dummy_function: eval(1) = " + std::to_string(r);
    } else {
        return "argument does NOT match dummy_function. This should never happen!";
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
    m.def("roundtrip", &roundtrip, py::arg("f"), py::arg("expect_none")=false);
    m.def("test_dummy_function", &test_dummy_function);
    // Export the payload constructor statistics for testing purposes:
    m.def("payload_cstats", &ConstructorStats::get<Payload>);
}
