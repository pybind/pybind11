/*
    tests/test_warnings.cpp -- usage of raise_warning() and warnings categories

    Copyright (c) 2023 Jan Iwaszkiewicz <jiwaszkiewicz6@gmail.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

#include <utility>

namespace warning_helpers {
void warn_function(py::module &m, const char *name, const PyObject *category, const char *message) {
    m.def(name, [category, message]() { py::raise_warning(message, category); });
}
} // namespace warning_helpers

class CustomWarning {};

TEST_SUBMODULE(warnings_, m) {

    // Test warning mechanism base
    m.def("raise_and_return", []() {
        std::string message = "Warning was raised!";
        py::raise_warning(message.c_str(), py::warnings::warning_base);
        return 21;
    });

    m.def("raise_default", []() { py::raise_warning("RuntimeWarning is raised!"); });

    m.def("raise_from_cpython",
          []() { py::raise_warning("UnicodeWarning is raised!", PyExc_UnicodeWarning); });

    m.def("raise_and_fail",
          []() { py::raise_warning("RuntimeError should be raised!", PyExc_Exception); });

    // Test custom warnings
    static py::warning<CustomWarning> my_warning(m, "CustomWarning", py::warnings::deprecation);

    m.def("raise_custom", []() {
        py::raise_warning("CustomWarning was raised!", my_warning);
        return 37;
    });

    m.def("raise_with_wrapper", []() {
        my_warning("This is raised from a wrapper.");
        return 42;
    });

    // Bind warning categories
    warning_helpers::warn_function(
        m, "raise_base_warning", py::warnings::warning_base, "This is Warning!");
    warning_helpers::warn_function(
        m, "raise_bytes_warning", py::warnings::bytes, "This is BytesWarning!");
    warning_helpers::warn_function(
        m, "raise_deprecation_warning", py::warnings::deprecation, "This is DeprecationWarning!");
    warning_helpers::warn_function(
        m, "raise_future_warning", py::warnings::future, "This is FutureWarning!");
    warning_helpers::warn_function(
        m, "raise_import_warning", py::warnings::import, "This is ImportWarning!");
    warning_helpers::warn_function(m,
                                   "raise_pending_deprecation_warning",
                                   py::warnings::pending_deprecation,
                                   "This is PendingDeprecationWarning!");
    warning_helpers::warn_function(
        m, "raise_resource_warning", py::warnings::resource, "This is ResourceWarning!");
    warning_helpers::warn_function(
        m, "raise_runtime_warning", py::warnings::runtime, "This is RuntimeWarning!");
    warning_helpers::warn_function(
        m, "raise_syntax_warning", py::warnings::syntax, "This is SyntaxWarning!");
    warning_helpers::warn_function(
        m, "raise_unicode_warning", py::warnings::unicode, "This is UnicodeWarning!");
    warning_helpers::warn_function(
        m, "raise_user_warning", py::warnings::user, "This is UserWarning!");
}
