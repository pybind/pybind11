/*
    tests/pybind11_tests.cpp -- pybind example plugin

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include "constructor_stats.h"

/*
For testing purposes, we define a static global variable here in a function that each individual
test .cpp calls with its initialization lambda.  It's convenient here because we can just not
compile some test files to disable/ignore some of the test code.

It is NOT recommended as a way to use pybind11 in practice, however: the initialization order will
be essentially random, which is okay for our test scripts (there are no dependencies between the
individual pybind11 test .cpp files), but most likely not what you want when using pybind11
productively.

Instead, see the "How can I reduce the build time?" question in the "Frequently asked questions"
section of the documentation for good practice on splitting binding code over multiple files.
*/
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
    py::module m("pybind11_tests", "pybind testing plugin");

    bind_ConstructorStats(m);

    for (const auto &initializer : initializers())
        initializer(m);

    if (!py::hasattr(m, "have_eigen")) m.attr("have_eigen") = false;

    return m.ptr();
}
