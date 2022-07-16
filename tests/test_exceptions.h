#pragma once
#include "pybind11_tests.h"

#include <stdexcept>

// shared exceptions for cross_module_tests

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NS_VISIBILITY(pybind11_tests))

class PYBIND11_EXPORT_EXCEPTION shared_exception : public pybind11::builtin_exception {
public:
    using builtin_exception::builtin_exception;
    explicit shared_exception() : shared_exception("") {}
    void set_error() const override { PyErr_SetString(PyExc_RuntimeError, what()); }
};

PYBIND11_NAMESPACE_END(pybind11_tests)
