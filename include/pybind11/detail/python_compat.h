#pragma once

#include "common.h"

#include <iostream>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

// Implemented in internal_pytypes.h as we can safely assume a pybind_function callable
inline PyObject *pybind_backup_impl_vectorcall(PyObject *callable,
                                               PyObject *const *args,
                                               size_t nargs,
                                               PyObject *kwnames);

#ifdef Py_TPFLAGS_HAVE_VECTORCALL_NO
// Actual vectorcall implementation

constexpr unsigned long pybind_vectorcall_flag = Py_TPFLAGS_HAVE_VECTORCALL;

inline void set_vectorcall_set_offset(PyTypeObject &type, size_t offset) {
    type.tp_vectorcall_offset = static_cast<Py_ssize_t>(offset);
}
using pybind_vectorcallfunc = vectorcallfunc;

constexpr auto pybind_vectorcall_call = PyVectorcall_Call;

constexpr size_t pybind_vectorcall_arguments_offset = PY_VECTORCALL_ARGUMENTS_OFFSET;

inline size_t pybind_vectorcall_nargs(size_t nargs_with_flag) {
    return PyVectorcall_NARGS(nargs_with_flag);
}

#else
// Old python fallback vectorcall

constexpr unsigned long pybind_vectorcall_flag = 0;

inline void set_vectorcall_set_offset(PyTypeObject &, size_t) {}

using pybind_vectorcallfunc = PyObject *(*) (PyObject *, PyObject *const *, size_t, PyObject *);

constexpr size_t pybind_vectorcall_arguments_offset
    = (static_cast<size_t>(1)) << (static_cast<size_t>(8 * sizeof(size_t) - 1));

inline size_t pybind_vectorcall_nargs(size_t n) { return n & ~pybind_vectorcall_arguments_offset; }

inline PyObject *pybind_vectorcall_call(PyObject *callable, PyObject *args, PyObject *kwargs) {
    Py_ssize_t num_pos_args = PyTuple_GET_SIZE(args);

    Py_ssize_t num_args = num_pos_args;
    if (kwargs != nullptr) {
        num_args += PyDict_Size(kwargs);
    }

    if (num_args > 1000) {
        printf("Going to overflow the stack, so abort now\n");
        abort();
    }

    PyObject **arg_data
        = (PyObject **) alloca(sizeof(PyObject *) * (static_cast<size_t>(num_args) + 1));
    arg_data = arg_data + 1;
    for (Py_ssize_t i = 0; i < num_pos_args; i++) {
        arg_data[static_cast<size_t>(i)] = PyTuple_GET_ITEM(args, i);
    }

    PyObject *kwnames = nullptr;
    if (kwargs != nullptr && PyDict_Size(kwargs) > 0) {
        kwnames = PyTuple_New(PyDict_Size(kwargs));
        if (kwnames == nullptr) {
            pybind11_fail("Could not allocate tuple to store arguments");
        }

        PyObject *key = nullptr;
        PyObject *value = nullptr;
        Py_ssize_t pos = 0;
        Py_ssize_t i = 0;

        while (PyDict_Next(kwargs, &pos, &key, &value) != 0) {
            PyTuple_SET_ITEM(kwnames, i, key);
            Py_INCREF(key);
            arg_data[static_cast<size_t>(num_pos_args + i)] = value;
            i++;
        }
    }

    auto args_with_flag = static_cast<size_t>(num_pos_args);
    args_with_flag |= pybind_vectorcall_arguments_offset;

    PyObject *result = pybind_backup_impl_vectorcall(callable, arg_data, args_with_flag, kwnames);
    if (kwnames != nullptr) {
        Py_DECREF(kwnames);
    }
    return result;
}

#endif

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
