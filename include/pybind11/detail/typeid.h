/*
    pybind11/detail/typeid.h: Compiler-independent access to type identifiers

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include <cstdio>
#include <cstdlib>

#if defined(__GNUG__)
#    if !defined(__has_include)
// All supported Clang versions provide __has_include, but GCC versions before 5 do not.
// Preserve the previous unconditional __GNUG__ behavior for those older, still-supported
// GCC versions (see PR #6145).
#        define PYBIND11_HAS_CXXABI_H
#    elif __has_include(<cxxabi.h>)
#        define PYBIND11_HAS_CXXABI_H
#    endif
#endif
#if defined(PYBIND11_HAS_CXXABI_H)
#    include <cxxabi.h>
#endif

#include "common.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

/// Erase all occurrences of a substring
inline void erase_all(std::string &string, const std::string &search) {
    for (size_t pos = 0;;) {
        pos = string.find(search, pos);
        if (pos == std::string::npos) {
            break;
        }
        string.erase(pos, search.length());
    }
}

void clean_type_id(std::string &name);

std::string clean_type_id(const char *typeid_name);

PYBIND11_NAMESPACE_END(detail)

/// Return a string representation of a C++ type
template <typename T>
std::string type_id() {
    return detail::clean_type_id(typeid(T).name());
}

#if defined(PYBIND11_OPAQUE_PYOBJECT)
// typeid() needs a complete type; PyObject is opaque under the abi3t stable ABI.
template <>
inline std::string type_id<PyObject>() {
    return "PyObject";
}
#endif

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

#ifndef PYBIND11_PRECOMPILED
#    include "typeid-inl.h" // IWYU pragma: export
#endif
