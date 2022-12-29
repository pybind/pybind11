/*
    tests/test_object.cpp -- test object tracking

    Copyright (c) 2022 Zhixiong Tang <dvorak4tzx@gmail.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

struct Beed {
};
struct Pod {
    Beed &beed() {
        return _beed;
    }
  private:
    Beed _beed;
};

TEST_SUBMODULE(object_, m) {
    namespace py = pybind11;
    using rvp = py::return_value_policy;
    py::class_<Beed>(m, "Beed").def(py::init<>());
    py::class_<Pod>(m, "Pod").def(py::init<>()).def("beed", &Pod::beed, rvp::reference_internal);
}
