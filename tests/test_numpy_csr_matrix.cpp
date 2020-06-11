/*
  tests/test_numpy_csr_matrix.cpp --

  Copyright (c) 2016 Philip Degean

  All rights reserved. Use of this source code is governed by a
  BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include <pybind11/numpy/csr_matrix.h>

namespace py = pybind11;

template <typename T>
using py_csr_t = py::csr_t<T, py::array::c_style | py::array::forcecast>;

template <typename T>
void accept_csr_matrix(py_csr_t<T> &/*csr*/){

}

TEST_SUBMODULE(numpy_csr_matrix, m) {
    try { py::module::import("numpy"); }
    catch (...) { return; }

    m.def("accept_csr_matrix", accept_csr_matrix<double>, "accept_csr_matrix");
}
