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

    // test_exceptions.py
    m.def("raise_runtime_error", []() { PyErr_SetString(PyExc_RuntimeError, "My runtime error"); throw py::error_already_set(); });
    m.def("raise_value_error", []() { PyErr_SetString(PyExc_ValueError, "My value error"); throw py::error_already_set(); });
    m.def("throw_pybind_value_error", []() { throw py::value_error("pybind11 value error"); });
    m.def("throw_pybind_type_error", []() { throw py::type_error("pybind11 type error"); });
    m.def("throw_stop_iteration", []() { throw py::stop_iteration(); });

}
