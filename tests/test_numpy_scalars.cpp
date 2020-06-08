/*
  tests/test_numpy_scalars.cpp -- strict NumPy scalars

  Copyright (c) 2020 Ivan Smirnov

  All rights reserved. Use of this source code is governed by a
  BSD-style license that can be found in the LICENSE file.
*/

#include <complex>
#include <cstdint>
#include <string>
#include <utility>

#include "pybind11_tests.h"
#include <pybind11/numpy.h>

namespace py = pybind11;

template<typename T, typename F>
void register_test(py::module& m, const char *name, F&& func) {
    m.def("test_numpy_scalars", [=](py::numpy_scalar<T> v) {
        return std::make_tuple(name, py::make_scalar(static_cast<T>(func(v.value))));
    }, py::arg("x"));
    m.def((std::string("test_") + name).c_str(), [=](py::numpy_scalar<T> v) {
        return std::make_tuple(name, py::make_scalar(static_cast<T>(func(v.value))));
    }, py::arg("x"));
}

template<typename T>
struct add {
    T x;
    add(T x) : x(x) {}
    T operator()(T y) const { return static_cast<T>(x + y); }
};

TEST_SUBMODULE(numpy_scalars, m) {
    try { py::module::import("numpy"); }
    catch (...) { return; }

    using cfloat = std::complex<float>;
    using cdouble = std::complex<double>;
    using clongdouble = std::complex<long double>;

    register_test<bool>(m, "bool", [](bool x) { return !x; });
    register_test<int8_t>(m, "int8", add<int8_t>(-8));
    register_test<int16_t>(m, "int16", add<int16_t>(-16));
    register_test<int32_t>(m, "int32", add<int32_t>(-32));
    register_test<int64_t>(m, "int64", add<int64_t>(-64));
    register_test<uint8_t>(m, "uint8", add<uint8_t>(8));
    register_test<uint16_t>(m, "uint16", add<uint16_t>(16));
    register_test<uint32_t>(m, "uint32", add<uint32_t>(32));
    register_test<uint64_t>(m, "uint64", add<uint64_t>(64));
    register_test<float>(m, "float32", add<float>(0.125f));
    register_test<double>(m, "float64", add<double>(0.25f));
    register_test<long double>(m, "longdouble", add<long double>(0.5L));
    register_test<cfloat>(m, "complex64", add<cfloat>({0, -0.125f}));
    register_test<cdouble>(m, "complex128", add<cdouble>({0, -0.25}));
    register_test<clongdouble>(m, "longcomplex", add<clongdouble>({0, -0.5L}));
}
