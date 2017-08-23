/*
    tests/test_globaliostream.cpp -- Usage of global scoped_output_redirect

    Copyright (c) 2017 Henry F. Schreiner

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/


#include <pybind11/iostream.h>
#include "pybind11_tests.h"
#include <iostream>

TEST_SUBMODULE(globaliostream, m) {
    // test_evals
    m.def("redirect_output", []() {
            return py::capsule(new py::scoped_output_redirect(
                std::cout, py::module::import("sys").attr("stdout")
                ),
                [](void *sor) {
                    // Pypy seems to call this twice if you call del for some reason
                    delete static_cast<py::scoped_output_redirect *>(sor);
                });
            });

    m.def("c_output", [](std::string msg) {
        std::cout << msg << std::flush;
    });
}
