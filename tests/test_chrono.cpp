/*
    tests/test_chrono.cpp -- test conversions to/from std::chrono types

    Copyright (c) 2016 Trent Houliston <trent@houliston.me> and
                       Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/


#include "pybind11_tests.h"
#include "constructor_stats.h"
#include <pybind11/chrono.h>

// Return the current time off the wall clock
std::chrono::system_clock::time_point test_chrono1() {
    return std::chrono::system_clock::now();
}

// Round trip the passed in system clock time
std::chrono::system_clock::time_point test_chrono2(std::chrono::system_clock::time_point t) {
    return t;
}

// Round trip the passed in duration
std::chrono::system_clock::duration test_chrono3(std::chrono::system_clock::duration d) {
    return d;
}

// Difference between two passed in time_points
std::chrono::system_clock::duration test_chrono4(std::chrono::system_clock::time_point a, std::chrono::system_clock::time_point b) {
    return a - b;
}

// Return the current time off the steady_clock
std::chrono::steady_clock::time_point test_chrono5() {
    return std::chrono::steady_clock::now();
}

// Round trip a steady clock timepoint
std::chrono::steady_clock::time_point test_chrono6(std::chrono::steady_clock::time_point t) {
    return t;
}

// Roundtrip a duration in microseconds from a float argument
std::chrono::microseconds test_chrono7(std::chrono::microseconds t) {
    return t;
}

test_initializer chrono([] (py::module &m) {
    m.def("test_chrono1", &test_chrono1);
    m.def("test_chrono2", &test_chrono2);
    m.def("test_chrono3", &test_chrono3);
    m.def("test_chrono4", &test_chrono4);
    m.def("test_chrono5", &test_chrono5);
    m.def("test_chrono6", &test_chrono6);
    m.def("test_chrono7", &test_chrono7);
});
