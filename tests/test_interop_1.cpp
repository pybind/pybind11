/*
    tests/test_interop_1.cpp -- cross-framework interoperability tests

    Copyright (c) 2025 Hudson River Trading LLC <opensource@hudson-trading.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

// Use an unrealistically large internals version to isolate the test_interop
// modules from each other and from the rest of the pybind11 tests
#define PYBIND11_INTERNALS_VERSION 100

#include <pybind11/pybind11.h>

#include "test_interop.h"

namespace py = pybind11;

PYBIND11_MODULE(test_interop_1, m, py::mod_gil_not_used()) {
    Shared::bind_funcs</*SmartHolder=*/false>(m);
    m.def("bind_types", [hm = py::handle(m)]() { Shared::bind_types</*SmartHolder=*/false>(hm); });

    m.def("throw_shared", [](int v) { throw SharedExc{v}; });

    struct Convertible {
        int value;
    };
    py::class_<Convertible>(m, "Convertible")
        .def(py::init([](const Shared &arg) { return Convertible{arg.value}; }))
        .def_readonly("value", &Convertible::value);
    py::implicitly_convertible<Shared, Convertible>();
    m.def("test_implicit", [](Convertible conv) { return conv.value; });
}
