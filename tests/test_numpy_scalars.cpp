/*
  tests/test_numpy_scalars.cpp -- strict NumPy scalars

  Copyright (c) 2020 Ivan Smirnov

  All rights reserved. Use of this source code is governed by a
  BSD-style license that can be found in the LICENSE file.
*/

#include <cstdint>
#include <string>
#include <utility>

#include "pybind11_tests.h"
#include <pybind11/numpy.h>

namespace py = pybind11;

template<typename T>
void register_test(py::module& m, const char *name) {
    m.def("test_numpy_scalars", [=](py::numpy_scalar<T> v) {
        return std::make_tuple(name, py::make_scalar(static_cast<T>(v.value + 1)));
    }, py::arg("x"));
    m.def((std::string("test_") + name).c_str(), [=](py::numpy_scalar<T> v) {
        return std::make_tuple(name, py::make_scalar(static_cast<T>(v.value + 1)));
    }, py::arg("x"));
}

TEST_SUBMODULE(numpy_scalars, m) {
    try { py::module::import("numpy"); }
    catch (...) { return; }

    register_test<bool>(m, "bool");
    register_test<int8_t>(m, "int8");
    register_test<int16_t>(m, "int16");
    register_test<int32_t>(m, "int32");
    register_test<int64_t>(m, "int64");
    register_test<uint8_t>(m, "uint8");
    register_test<uint16_t>(m, "uint16");
    register_test<uint32_t>(m, "uint32");
    register_test<uint64_t>(m, "uint64");
    register_test<float>(m, "float32");
    register_test<double>(m, "float64");
}
