/*
    tests/test_warnings.cpp -- usage of warnings::warn() and warnings categories.

    Copyright (c) 2024 Jan Iwaszkiewicz <jiwaszkiewicz6@gmail.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <pybind11/warnings.h>

#include "pybind11_tests.h"

#include <utility>

TEST_SUBMODULE(warnings_, m) {

    // Test warning mechanism base
    m.def("warn_and_return_value", []() {
        std::string message = "This is simple warning";
        py::warnings::warn(message.c_str(), PyExc_Warning);
        return 21;
    });

    m.def("warn_with_default_category", []() { py::warnings::warn("This is RuntimeWarning"); });

    m.def("warn_with_different_category",
          []() { py::warnings::warn("This is FutureWarning", PyExc_FutureWarning); });

    m.def("warn_with_invalid_category",
          []() { py::warnings::warn("Invalid category", PyExc_Exception); });

    // Test custom warnings
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object> ex_storage;
    ex_storage.call_once_and_store_result([&]() {
        return py::warnings::new_warning_type(m, "CustomWarning", PyExc_DeprecationWarning);
    });

    m.def("warn_with_custom_type", []() {
        py::warnings::warn("This is CustomWarning", ex_storage.get_stored());
        return 37;
    });

    m.def("register_duplicate_warning",
          [m]() { py::warnings::new_warning_type(m, "CustomWarning", PyExc_RuntimeWarning); });
}
