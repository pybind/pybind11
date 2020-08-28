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
#include <cxxabi.h>
#endif

#include "common.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)
/// Erase all occurrences of a substring
void erase_all(std::string &string, const std::string &search);

PYBIND11_NOINLINE void clean_type_id(std::string &name);
PYBIND11_NAMESPACE_END(detail)

/// Return a string representation of a C++ type
template <typename T> static std::string type_id() {
    std::string name(typeid(T).name());
    detail::clean_type_id(name);
    return name;
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

#if !defined(PYBIND11_DECLARATIONS_ONLY)
#include "typeid-inl.h"
#endif
