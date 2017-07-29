/*
    tests/pybind11_cross_module_tests.cpp -- contains tests that require multiple modules

    Copyright (c) 2017 Jason Rhinelander <jason@imaginary.ca>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

PYBIND11_MODULE(pybind11_cross_module_tests, m) {
    m.doc() = "pybind11 cross-module test module";

    // test_local_bindings.py tests:
    //
    // Definitions here are tested by importing both this module and the
    // relevant pybind11_tests submodule from a test_whatever.py

}
