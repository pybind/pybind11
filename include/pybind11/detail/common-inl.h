#include "common.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

PYBIND11_INLINE void pybind11_fail(const char *reason) { throw std::runtime_error(reason); }

PYBIND11_INLINE void pybind11_fail(const std::string &reason) { throw std::runtime_error(reason); }

PYBIND11_INLINE error_scope::error_scope() { PyErr_Fetch(&type, &value, &trace); }

PYBIND11_INLINE error_scope::~error_scope() { PyErr_Restore(type, value, trace); }

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
