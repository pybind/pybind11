/*
    tests/test_thread.cpp -- call pybind11 bound methods in threads

    Copyright (c) 2017 Laramie Leavitt (Google LLC) <lar@google.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <string_view>
#include <string>

#define PYBIND11_HAS_STRING_VIEW 1

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pybind11_tests.h"


TEST_SUBMODULE(thread, m) {

    // std::string_view uses loader_life_support to ensure that the string contents
    // remains alive for the life of the call. These methods are invoked concurrently
    m.def("method", [](std::string_view str) -> std::string {
        return std::string(str);
    });

    m.def("method_no_gil", [](std::string_view str) -> std::string {
        return std::string(str);
    },
    py::call_guard<py::gil_scoped_release>());

}
