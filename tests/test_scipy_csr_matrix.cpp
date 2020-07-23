/*
  tests/test_numpy_csr_matrix.cpp --

  Copyright (c) 2016 Philip Degean

  All rights reserved. Use of this source code is governed by a
  BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include <pybind11/scipy.h>

namespace py = pybind11;

template <typename T>
using py_csr_t = py::csr_t<T, py::array::c_style | py::array::forcecast>;

template <typename T>
void swap_first_last_data(py_csr_t<T> &/*csr*/){

}

TEST_SUBMODULE(numpy_csr_matrix, m) {
    try { py::module::import("scipy"); }
    catch (...) { return; }

    m.def("swap_first_last_data", swap_first_last_data<double>, "swap_first_last_data");
}
