/*
    pybind11/detail/exception_translation-inl.h: Out-of-line definitions for
   exception_translation.h

    Copyright (c) 2024 The Pybind Development Team.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

// Every function defined here must start with PYBIND11_INLINE (or
// PYBIND11_NOINLINE_ATTR PYBIND11_INLINE). In the default header-only mode this file is
// included at the bottom of exception_translation.h; when PYBIND11_PRECOMPILED is defined
// it is only compiled into the pybind11 static library (see src/).

#pragma once

#include "exception_translation.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

PYBIND11_INLINE bool
apply_exception_translators(std::forward_list<ExceptionTranslator> &translators) {
    auto last_exception = std::current_exception();

    for (auto &translator : translators) {
        try {
            translator(last_exception);
            return true;
        } catch (...) {
            last_exception = std::current_exception();
        }
    }
    return false;
}

PYBIND11_INLINE void try_translate_exceptions() {
    /* When an exception is caught, give each registered exception
        translator a chance to translate it to a Python exception. First
        all module-local translators will be tried in reverse order of
        registration. If none of the module-locale translators handle
        the exception (or there are no module-locale translators) then
        the global translators will be tried, also in reverse order of
        registration.

        A translator may choose to do one of the following:

        - catch the exception and call py::set_error()
            to set a standard (or custom) Python exception, or
        - do nothing and let the exception fall through to the next translator, or
        - delegate translation to the next translator by throwing a new type of exception.
        */

    bool handled = with_exception_translators(
        [&](std::forward_list<ExceptionTranslator> &exception_translators,
            std::forward_list<ExceptionTranslator> &local_exception_translators) {
            if (detail::apply_exception_translators(local_exception_translators)) {
                return true;
            }
            if (detail::apply_exception_translators(exception_translators)) {
                return true;
            }
            return false;
        });

    if (!handled) {
        set_error(PyExc_SystemError, "Exception escaped from default exception translator!");
    }
}

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
