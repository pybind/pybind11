/*
    tests/pybind11_cross_module_tests.cpp -- contains tests that require multiple modules

    Copyright (c) 2017 Jason Rhinelander <jason@imaginary.ca>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include "local_bindings.h"
#include <pybind11/stl_bind.h>

PYBIND11_MODULE(pybind11_cross_module_tests, m) {
    m.doc() = "pybind11 cross-module test module";

    // test_local_bindings.py tests:
    //
    // Definitions here are tested by importing both this module and the
    // relevant pybind11_tests submodule from a test_whatever.py

    // test_exceptions.py
    m.def("raise_runtime_error", []() { PyErr_SetString(PyExc_RuntimeError, "My runtime error"); throw py::error_already_set(); });
    m.def("raise_value_error", []() { PyErr_SetString(PyExc_ValueError, "My value error"); throw py::error_already_set(); });
    m.def("throw_pybind_value_error", []() { throw py::value_error("pybind11 value error"); });
    m.def("throw_pybind_type_error", []() { throw py::type_error("pybind11 type error"); });
    m.def("throw_stop_iteration", []() { throw py::stop_iteration(); });

    // test_local_bindings.py
    // Local to both:
    bind_local<LocalType, 1>(m, "LocalType", py::module_local())
        .def("get2", [](LocalType &t) { return t.i + 2; })
        ;

    // Can only be called with our python type:
    m.def("local_value", [](LocalType &l) { return l.i; });

    // test_nonlocal_failure
    // This registration will fail (global registration when LocalFail is already registered
    // globally in the main test module):
    m.def("register_nonlocal", [m]() {
        bind_local<NonLocalType, 0>(m, "NonLocalType");
    });

    // test_stl_bind_local
    // stl_bind.h binders defaults to py::module_local if the types are local or converting:
    py::bind_vector<std::vector<LocalType>>(m, "LocalVec");
    py::bind_map<std::unordered_map<std::string, LocalType>>(m, "LocalMap");
    // and global if the type (or one of the types, for the map) is global (so these will fail,
    // assuming pybind11_tests is already loaded):
    m.def("register_nonlocal_vec", [m]() {
        py::bind_vector<std::vector<NonLocalType>>(m, "NonLocalVec");
    });
    m.def("register_nonlocal_map", [m]() {
        py::bind_map<std::unordered_map<std::string, NonLocalType>>(m, "NonLocalMap");
    });

    // test_stl_bind_global
    // The default can, however, be overridden to global using `py::module_local()` or
    // `py::module_local(false)`.
    // Explicitly made local:
    py::bind_vector<std::vector<NonLocal2>>(m, "NonLocalVec2", py::module_local());
    // Explicitly made global (and so will fail to bind):
    m.def("register_nonlocal_map2", [m]() {
        py::bind_map<std::unordered_map<std::string, uint8_t>>(m, "NonLocalMap2", py::module_local(false));
    });

    // test_mixed_local_global
    // We try this both with the global type registered first and vice versa (the order shouldn't
    // matter).
    m.def("register_mixed_global_local", [m]() {
        bind_local<MixedGlobalLocal, 200>(m, "MixedGlobalLocal", py::module_local());
    });
    m.def("register_mixed_local_global", [m]() {
        bind_local<MixedLocalGlobal, 2000>(m, "MixedLocalGlobal", py::module_local(false));
    });
    m.def("get_mixed_gl", [](int i) { return MixedGlobalLocal(i); });
    m.def("get_mixed_lg", [](int i) { return MixedLocalGlobal(i); });

    // test_internal_locals_differ
    m.def("local_cpp_types_addr", []() { return (uintptr_t) &py::detail::registered_local_types_cpp(); });
}
