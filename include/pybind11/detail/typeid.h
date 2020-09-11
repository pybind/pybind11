/*
    pybind11/detail/typeid.h: Compiler-independent access to type identifiers

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include <cstdlib>
#include <memory>
#include <string>
#include <typeinfo>

#include "pybind11/attr.h"
#include "pybind11/cast.h"
#include "pybind11/chrono.h"
#include "pybind11/complex.h"
#include "pybind11/detail/class.h"
#include "pybind11/detail/common.h"
#include "pybind11/detail/init.h"
#include "pybind11/embed.h"
#include "pybind11/eval.h"
#include "pybind11/functional.h"
#include "pybind11/iostream.h"
#include "pybind11/numpy.h"
#include "pybind11/operators.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"
#include "third_party/stl/itanium_abi/cxxabi.h"

#if defined(__GNUG__)
#include <cxxabi.h>
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)
/// Erase all occurrences of a substring
inline void erase_all(std::string &string, const std::string &search) {
    for (size_t pos = 0;;) {
        pos = string.find(search, pos);
        if (pos == std::string::npos) break;
        string.erase(pos, search.length());
    }
}

PYBIND11_NOINLINE inline void clean_type_id(std::string &name) {
#if defined(__GNUG__)
    int status = 0;
    std::unique_ptr<char, void (*)(void *)> res {
        abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status), std::free };
    if (status == 0)
        name = res.get();
#else
    detail::erase_all(name, "class ");
    detail::erase_all(name, "struct ");
    detail::erase_all(name, "enum ");
#endif
    detail::erase_all(name, "pybind11::");
}
PYBIND11_NAMESPACE_END(detail)

/// Return a string representation of a C++ type
template <typename T> static std::string type_id() {
    std::string name(typeid(T).name());
    detail::clean_type_id(name);
    return name;
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
