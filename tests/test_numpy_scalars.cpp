/*
  tests/test_numpy_scalars.cpp -- strict NumPy scalars

  Copyright (c) 2021 Steve R. Sun

  All rights reserved. Use of this source code is governed by a
  BSD-style license that can be found in the LICENSE file.
*/

#include <pybind11/numpy.h>

#include "pybind11_tests.h"

#include <complex>
#include <cstdint>

namespace py = pybind11;

namespace pybind11_test_numpy_scalars {

template <typename T>
struct add {
    T x;
    explicit add(T x) : x(x) {}
    T operator()(T y) const { return static_cast<T>(x + y); }
};

template <typename T, typename F>
void register_test(py::module &m, const char *name, F &&func) {
    m.def((std::string("test_") + name).c_str(),
          [=](py::numpy_scalar<T> v) {
              return std::make_tuple(name, py::make_scalar(static_cast<T>(func(v.value))));
          },
          py::arg("x"));
}

} // namespace pybind11_test_numpy_scalars

using namespace pybind11_test_numpy_scalars;

TEST_SUBMODULE(numpy_scalars, m) {
    using cfloat = std::complex<float>;
    using cdouble = std::complex<double>;

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
    register_test<cfloat>(m, "complex64", add<cfloat>({0, -0.125f}));
    register_test<cdouble>(m, "complex128", add<cdouble>({0, -0.25f}));

    m.def("test_eq",
          [](py::numpy_scalar<int32_t> a, py::numpy_scalar<int32_t> b) { return a == b; });
    m.def("test_ne",
          [](py::numpy_scalar<int32_t> a, py::numpy_scalar<int32_t> b) { return a != b; });
}
