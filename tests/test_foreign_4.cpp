/*
    tests/test_foreign_4.cpp -- cross-framework interoperability tests
    (on_request mode: no automatic import or export)

    Copyright (c) 2025 Hudson River Trading LLC <opensource@hudson-trading.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

// Use an unrealistically large internals version to isolate the test_foreign
// modules from each other and from the rest of the pybind11 tests
#if defined(PYBIND11_INTERNALS_VERSION)
#    undef PYBIND11_INTERNALS_VERSION
#endif
#define PYBIND11_INTERNALS_VERSION 400

#include <pybind11/pybind11.h>

#include "test_foreign.h"

namespace py = pybind11;

PYBIND11_MODULE(test_foreign_4, m, py::mod_gil_not_used(), py::foreign_interop::on_request()) {
    py::handle hm = m;
    Shared::bind_funcs</*SmartHolder=*/true>(m);
    m.def("bind_types", [hm]() { Shared::bind_types</*SmartHolder=*/true>(hm); });
}
