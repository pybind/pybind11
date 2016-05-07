/*
    example/example.cpp -- pybind example plugin

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"

void init_ex1(py::module &);
void init_ex2(py::module &);
void init_ex3(py::module &);
void init_ex4(py::module &);
void init_ex5(py::module &);
void init_ex6(py::module &);
void init_ex7(py::module &);
void init_ex8(py::module &);
void init_ex9(py::module &);
void init_ex10(py::module &);
void init_ex11(py::module &);
void init_ex12(py::module &);
void init_ex13(py::module &);
void init_ex14(py::module &);
void init_ex15(py::module &);
void init_ex16(py::module &);
void init_ex17(py::module &);
void init_issues(py::module &);

#if defined(PYBIND11_TEST_EIGEN)
    void init_eigen(py::module &);
#endif

PYBIND11_PLUGIN(example) {
    py::module m("example", "pybind example plugin");

    init_ex1(m);
    init_ex2(m);
    init_ex3(m);
    init_ex4(m);
    init_ex5(m);
    init_ex6(m);
    init_ex7(m);
    init_ex8(m);
    init_ex9(m);
    init_ex10(m);
    init_ex11(m);
    init_ex12(m);
    init_ex13(m);
    init_ex14(m);
    init_ex15(m);
    init_ex16(m);
    init_ex17(m);
    init_issues(m);

    #if defined(PYBIND11_TEST_EIGEN)
        init_eigen(m);
    #endif

    return m.ptr();
}
