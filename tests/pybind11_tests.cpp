/*
    tests/pybind11_tests.cpp -- pybind example plugin

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include "constructor_stats.h"

std::list<std::function<void(py::module &)>> &initializers() {
    static std::list<std::function<void(py::module &)>> inits;
    return inits;
}

test_initializer::test_initializer(std::function<void(py::module &)> initializer) {
    initializers().push_back(std::move(initializer));
}

void bind_ConstructorStats(py::module &m) {
    py::class_<ConstructorStats>(m, "ConstructorStats")
        .def("alive", &ConstructorStats::alive)
        .def("values", &ConstructorStats::values)
        .def_readwrite("default_constructions", &ConstructorStats::default_constructions)
        .def_readwrite("copy_assignments", &ConstructorStats::copy_assignments)
        .def_readwrite("move_assignments", &ConstructorStats::move_assignments)
        .def_readwrite("copy_constructions", &ConstructorStats::copy_constructions)
        .def_readwrite("move_constructions", &ConstructorStats::move_constructions)
        .def_static("get", (ConstructorStats &(*)(py::object)) &ConstructorStats::get, py::return_value_policy::reference_internal);
}

PYBIND11_PLUGIN(pybind11_tests) {
    py::module m("pybind11_tests", "pybind example plugin");

    bind_ConstructorStats(m);

    for (const auto &initializer : initializers())
        initializer(m);

    if (!py::hasattr(m, "have_eigen")) m.attr("have_eigen") = false;

    return m.ptr();
}
