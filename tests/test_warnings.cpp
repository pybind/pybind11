/*
    tests/test_warnings.cpp -- usage of warnings::warn() and warnings categories.

    Copyright (c) 2024 Jan Iwaszkiewicz <jiwaszkiewicz6@gmail.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <pybind11/warnings.h>

#include "pybind11_tests.h"

#include <utility>

namespace warning_helpers {
void warn_function(py::module &m, const char *name, py::handle category, const char *message) {
    m.def(name, [category, message]() { py::warnings::warn(message, category); });
}
} // namespace warning_helpers

class CustomWarning {};

TEST_SUBMODULE(warnings_, m) {

    // Test warning mechanism base
    m.def("raise_and_return", []() {
        std::string message = "Warning was raised!";
        py::warnings::warn(message.c_str(), PyExc_Warning);
        return 21;
    });

    m.def("raise_default", []() { py::warnings::warn("RuntimeWarning is raised!"); });

    m.def("raise_from_cpython",
          []() { py::warnings::warn("UnicodeWarning is raised!", PyExc_UnicodeWarning); });

    m.def("raise_and_fail",
          []() { py::warnings::warn("RuntimeError should be raised!", PyExc_Exception); });

    // Test custom warnings
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object> ex_storage;
    ex_storage.call_once_and_store_result([&]() {
        return py::warnings::new_warning_type(m, "CustomWarning", PyExc_DeprecationWarning);
    });

    m.def("raise_custom", []() {
        py::warnings::warn("CustomWarning was raised!", ex_storage.get_stored());
        return 37;
    });

    m.def("register_duplicate_warning",
          [m]() { py::warnings::new_warning_type(m, "CustomWarning", PyExc_RuntimeWarning); });

    // Bind warning categories
    warning_helpers::warn_function(m, "raise_base_warning", PyExc_Warning, "This is Warning!");
    warning_helpers::warn_function(
        m, "raise_bytes_warning", PyExc_BytesWarning, "This is BytesWarning!");
    warning_helpers::warn_function(
        m, "raise_deprecation_warning", PyExc_DeprecationWarning, "This is DeprecationWarning!");
    warning_helpers::warn_function(
        m, "raise_future_warning", PyExc_FutureWarning, "This is FutureWarning!");
    warning_helpers::warn_function(
        m, "raise_import_warning", PyExc_ImportWarning, "This is ImportWarning!");
    warning_helpers::warn_function(m,
                                   "raise_pending_deprecation_warning",
                                   PyExc_PendingDeprecationWarning,
                                   "This is PendingDeprecationWarning!");
    warning_helpers::warn_function(
        m, "raise_resource_warning", PyExc_ResourceWarning, "This is ResourceWarning!");
    warning_helpers::warn_function(
        m, "raise_runtime_warning", PyExc_RuntimeWarning, "This is RuntimeWarning!");
    warning_helpers::warn_function(
        m, "raise_syntax_warning", PyExc_SyntaxWarning, "This is SyntaxWarning!");
    warning_helpers::warn_function(
        m, "raise_unicode_warning", PyExc_UnicodeWarning, "This is UnicodeWarning!");
    warning_helpers::warn_function(
        m, "raise_user_warning", PyExc_UserWarning, "This is UserWarning!");
}
