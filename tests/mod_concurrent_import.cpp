// Copyright (c) 2026 The pybind Community.

#include <pybind11/pybind11.h>

namespace py = pybind11;

// test_exceptions.py needs a module whose *first* import happens inside a test, so nothing else
// may import this one.

PYBIND11_MODULE(mod_concurrent_import, , py::mod_gil_not_used()) {}
