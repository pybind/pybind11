/*
    tests/test_iostream.cpp -- Usage of scoped_output_redirect

    Copyright (c) 2017 Henry F. Schreiner

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/


#include <pybind11/iostream.h>
#include "pybind11_tests.h"
#include <iostream>

TEST_SUBMODULE(iostream, m) {
    // test_evals

    m.def("captured_output", [](std::string msg) {
        py::scoped_output_redirect redir(std::cout, py::module::import("sys").attr("stdout"));
        std::cout << msg << std::flush;
    });

    m.def("raw_output", [](std::string msg) {
        std::cout << msg << std::flush;
    });

    m.def("captured_dual", [](std::string msg, std::string emsg) {
        py::scoped_output_redirect redirout(std::cout, py::module::import("sys").attr("stdout"));
        py::scoped_output_redirect redirerr(std::cerr, py::module::import("sys").attr("stderr"));
        std::cout << msg << std::flush;
        std::cerr << emsg << std::flush;
    });

}
