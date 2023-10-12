/*
    tests/cross_module_gil_utils.cpp -- tools for acquiring GIL from a different module

    Copyright (c) 2019 Google LLC

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/
#include <pybind11/pybind11.h>

#include <cstdint>

// This file mimics a DSO that makes pybind11 calls but does not define a
// PYBIND11_MODULE. The purpose is to test that such a DSO can create a
// py::gil_scoped_acquire when the running thread is in a GIL-released state.
//
// Note that we define a Python module here for convenience, but in general
// this need not be the case. The typical scenario would be a DSO that implements
// shared logic used internally by multiple pybind11 modules.

namespace {

namespace py = pybind11;
void gil_acquire() { py::gil_scoped_acquire gil; }

constexpr char kModuleName[] = "cross_module_gil_utils";

#if PY_MAJOR_VERSION >= 3
struct PyModuleDef moduledef
    = {PyModuleDef_HEAD_INIT, kModuleName, NULL, 0, NULL, NULL, NULL, NULL, NULL};
#else
PyMethodDef module_methods[] = {{NULL, NULL, 0, NULL}};
#endif

} // namespace

extern "C" PYBIND11_EXPORT
#if PY_MAJOR_VERSION >= 3
    PyObject *
    PyInit_cross_module_gil_utils()
#else
    void
    initcross_module_gil_utils()
#endif
{

    PyObject *m =
#if PY_MAJOR_VERSION >= 3
        PyModule_Create(&moduledef);
#else
        Py_InitModule(kModuleName, module_methods);
#endif

    if (m != NULL) {
        static_assert(sizeof(&gil_acquire) == sizeof(void *),
                      "Function pointer must have the same size as void*");
        PyModule_AddObject(
            m, "gil_acquire_funcaddr", PyLong_FromVoidPtr(reinterpret_cast<void *>(&gil_acquire)));
    }

#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}
