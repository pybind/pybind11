/*
    example/issues.cpp -- collection of testcases for miscellaneous issues

    Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"


void init_issues(py::module &m) {
    py::module m2 = m.def_submodule("issues");

    // #137: const char* isn't handled properly
    m2.def("print_cchar", [](const char *string) { std::cout << string << std::endl; });
}
