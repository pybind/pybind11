/*
    tests/pybind11_tests.cpp -- pybind example plugin

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include "constructor_stats.h"

void init_ex_methods_and_attributes(py::module &);
void init_ex_python_types(py::module &);
void init_ex_operator_overloading(py::module &);
void init_ex_constants_and_functions(py::module &);
void init_ex_callbacks(py::module &);
void init_ex_sequences_and_iterators(py::module &);
void init_ex_buffers(py::module &);
void init_ex_smart_ptr(py::module &);
void init_ex_modules(py::module &);
void init_ex_numpy_vectorize(py::module &);
void init_ex_arg_keywords_and_defaults(py::module &);
void init_ex_virtual_functions(py::module &);
void init_ex_keep_alive(py::module &);
void init_ex_opaque_types(py::module &);
void init_ex_pickling(py::module &);
void init_ex_inheritance(py::module &);
void init_ex_stl_binder_vector(py::module &);
void init_ex_eval(py::module &);
void init_ex_custom_exceptions(py::module &);
void init_ex_numpy_dtypes(py::module &);
void init_ex_enum(py::module &);
void init_issues(py::module &);

#if defined(PYBIND11_TEST_EIGEN)
    void init_eigen(py::module &);
#endif

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

    init_ex_methods_and_attributes(m);
    init_ex_python_types(m);
    init_ex_operator_overloading(m);
    init_ex_constants_and_functions(m);
    init_ex_callbacks(m);
    init_ex_sequences_and_iterators(m);
    init_ex_buffers(m);
    init_ex_smart_ptr(m);
    init_ex_modules(m);
    init_ex_numpy_vectorize(m);
    init_ex_arg_keywords_and_defaults(m);
    init_ex_virtual_functions(m);
    init_ex_keep_alive(m);
    init_ex_opaque_types(m);
    init_ex_pickling(m);
    init_ex_inheritance(m);
    init_ex_stl_binder_vector(m);
    init_ex_eval(m);
    init_ex_custom_exceptions(m);
    init_ex_numpy_dtypes(m);
    init_ex_enum(m);
    init_issues(m);

#if defined(PYBIND11_TEST_EIGEN)
    init_eigen(m);
    m.attr("have_eigen") = py::cast(true);
#else
    m.attr("have_eigen") = py::cast(false);
#endif

    return m.ptr();
}
