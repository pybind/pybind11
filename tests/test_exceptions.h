#pragma once
#include "pybind11_tests.h"
#include <stdexcept>

// shared exceptions for cross_module_tests

class PYBIND11_EXPORT tmp_e : public pybind11::builtin_exception {
public:
    using builtin_exception::builtin_exception;
    explicit tmp_e() : tmp_e("") {}
    void set_error() const override { PyErr_SetString(PyExc_RuntimeError, what()); }
};
