/*
    pybind11/detail/typeid-inl.h: Out-of-line definitions for typeid.h

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

// Every function defined here must start with PYBIND11_INLINE (or
// PYBIND11_NOINLINE_ATTR PYBIND11_INLINE). In the default header-only mode this file is
// included at the bottom of typeid.h; when PYBIND11_PRECOMPILED is defined it is only
// compiled into the pybind11 static library (see src/).

#pragma once

#include "typeid.h"

#include <cstdlib>
#include <memory>
#include <string>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

PYBIND11_NOINLINE_ATTR PYBIND11_INLINE void clean_type_id(std::string &name) {
#if defined(PYBIND11_HAS_CXXABI_H)
    int status = 0;
    std::unique_ptr<char, void (*)(void *)> res{
        abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status), std::free};
    if (status == 0) {
        name = res.get();
    }
#else
    detail::erase_all(name, "class ");
    detail::erase_all(name, "struct ");
    detail::erase_all(name, "enum ");
#endif
    detail::erase_all(name, "pybind11::");
}

PYBIND11_INLINE std::string clean_type_id(const char *typeid_name) {
    std::string name(typeid_name);
    detail::clean_type_id(name);
    return name;
}

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
