/*
    tests/test_interop_2.cpp -- cross-framework interoperability tests

    Copyright (c) 2025 Hudson River Trading LLC <opensource@hudson-trading.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

// Use an unrealistically large internals version to isolate the test_interop
// modules from each other and from the rest of the pybind11 tests
#define PYBIND11_INTERNALS_VERSION 200

#include <pybind11/pybind11.h>

#include "test_interop.h"

namespace py = pybind11;

PYBIND11_MODULE(test_interop_2, m, py::mod_gil_not_used()) {
    Shared::bind_funcs</*SmartHolder=*/true>(m);
    m.def("bind_types", [hm = py::handle(m)]() { Shared::bind_types</*SmartHolder=*/true>(hm); });
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            std::rethrow_exception(p);
        } catch (const SharedExc &s) {
            // Instead of just calling PyErr_SetString, exercise the
            // path where one translator throws an exception to be handled
            // by another.
            throw py::value_error(std::string(py::str("Shared({})").format(s.value)).c_str());
        }
    });
    m.def("throw_shared", [](int v) { throw SharedExc{v}; });
}
