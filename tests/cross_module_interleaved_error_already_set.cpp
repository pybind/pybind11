/*
    Copyright (c) 2022 Google LLC

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <pybind11/pybind11.h>

// This file mimics a DSO that makes pybind11 calls but does not define a PYBIND11_MODULE,
// so that the first call of cross_module_error_already_set() triggers the first call of
// pybind11::detail::get_internals().

namespace {

namespace py = pybind11;

void interleaved_error_already_set() {
    py::set_error(PyExc_RuntimeError, "1st error.");
    try {
        throw py::error_already_set();
    } catch (const py::error_already_set &) {
        // The 2nd error could be conditional in a real application.
        py::set_error(PyExc_RuntimeError, "2nd error.");
    } // Here the 1st error is destroyed before the 2nd error is fetched.
    // The error_already_set dtor triggers a pybind11::detail::get_internals()
    // call via pybind11::gil_scoped_acquire.
    if (PyErr_Occurred()) {
        throw py::error_already_set();
    }
}

constexpr char kModuleName[] = "cross_module_interleaved_error_already_set";

#if !defined(PYBIND11_OPAQUE_PYOBJECT)
struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, kModuleName, nullptr, 0, nullptr, nullptr, nullptr, nullptr, nullptr};
#endif

int module_exec(PyObject *m) {
    static_assert(sizeof(&interleaved_error_already_set) == sizeof(void *),
                  "Function pointer must have the same size as void *");
    return PyModule_AddObject(
        m,
        "funcaddr",
        PyLong_FromVoidPtr(reinterpret_cast<void *>(&interleaved_error_already_set)));
}

} // namespace

#if defined(PYBIND11_OPAQUE_PYOBJECT)
// PEP 793 export hook: PyModuleDef is an incomplete type under the abi3t stable ABI.
extern "C" PYBIND11_EXPORT PySlot *PyModExport_cross_module_interleaved_error_already_set() {
    PyABIInfo_VAR(abi_info);
    static PySlot slots[] = {PySlot_PTR(Py_mod_name, kModuleName),
                             {Py_mod_exec, 0, {0}, {reinterpret_cast<void *>(&module_exec)}},
                             PySlot_PTR(Py_mod_gil, Py_MOD_GIL_NOT_USED),
                             PySlot_PTR_STATIC(Py_mod_abi, &abi_info),
                             {0, 0, {0}, {nullptr}}};
    return slots;
}
#else
extern "C" PYBIND11_EXPORT PyObject *PyInit_cross_module_interleaved_error_already_set() {
    PyObject *m = PyModule_Create(&moduledef);
    if (m != nullptr) {
#    ifdef Py_GIL_DISABLED
        PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#    endif
        if (module_exec(m) != 0) {
            Py_DECREF(m);
            return nullptr;
        }
    }
    return m;
}
#endif
