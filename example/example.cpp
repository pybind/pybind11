/*
    example/example.cpp -- pybind example plugin

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"

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
void init_issues(py::module &);

#if defined(PYBIND11_TEST_EIGEN)
    void init_eigen(py::module &);
#endif

PYBIND11_PLUGIN(example) {
    py::module m("example", "pybind example plugin");

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
    init_issues(m);

    #if defined(PYBIND11_TEST_EIGEN)
        init_eigen(m);
    #endif

    return m.ptr();
}
