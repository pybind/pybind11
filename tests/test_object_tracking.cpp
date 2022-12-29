/*
    tests/test_object_tracking.cpp -- test object tracking

    Copyright (c) 2022 Zhixiong Tang <dvorak4tzx@gmail.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

struct Bead {};
struct Pod {
    Bead &bead() { return _bead; }

private:
    Bead _bead;
};

TEST_SUBMODULE(object_, m) {
    namespace py = pybind11;
    using rvp = py::return_value_policy;
    py::class_<Bead>(m, "Bead").def(py::init<>());
    py::class_<Pod>(m, "Pod").def(py::init<>()).def("bead", &Pod::bead, rvp::reference_internal);
}
