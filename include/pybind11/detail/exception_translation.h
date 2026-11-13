/*
    pybind11/detail/exception_translation.h: means to translate C++ exceptions to Python exceptions

    Copyright (c) 2024 The Pybind Development Team.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"
#include "internals.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

// Apply all the extensions translators from a list
// Return true if one of the translators completed without raising an exception
// itself. Return of false indicates that if there are other translators
// available, they should be tried.
bool apply_exception_translators(std::forward_list<ExceptionTranslator> &translators);

void try_translate_exceptions();

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

#ifndef PYBIND11_PRECOMPILED
#    include "exception_translation-inl.h" // IWYU pragma: export
#endif
