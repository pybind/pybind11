/*
    tests/test_foreign_5.cpp -- cross-framework interoperability tests
    (disabled mode: foreign interop completely disabled)

    Copyright (c) 2025 Hudson River Trading LLC <opensource@hudson-trading.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

// Use an unrealistically large internals version to isolate the test_foreign
// modules from each other and from the rest of the pybind11 tests
#define PYBIND11_INTERNALS_VERSION 500

#include <pybind11/pybind11.h>

#include "test_foreign.h"

namespace py = pybind11;

PYBIND11_MODULE(test_foreign_5, m, py::mod_gil_not_used(), py::foreign_interop::disabled()) {
    py::handle hm = m;
    Shared::bind_funcs</*SmartHolder=*/true>(m);
    m.def("bind_types", [hm]() { Shared::bind_types</*SmartHolder=*/true>(hm); });
}
