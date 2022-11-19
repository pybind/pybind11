// Copyright (c) 2022 The pybind Community.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include "common.h"

#include <functional>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

/* NOTE - ATTENTION - WARNING - EXTREME CAUTION

   Almost any changes here will break compatibility with `PYBIND11_INTERNALS_VERSION 4`

   Recommendation:
       To not break compatibility with many existing PyPI wheels (https://pypi.org/),
                            DO NOT MAKE CHANGES HERE
       until Python 3.11 (at least) has reached EOL. Then remove this file entirely.

   To evolve this code, start with a copy of this file and move the code to
       namespace pybind11::cross_extension_shared_states
   with new, versioned names, e.g. type_map_v2.
 */

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

// Python loads modules by default with dlopen with the RTLD_LOCAL flag; under libc++ and possibly
// other STLs, this means `typeid(A)` from one module won't equal `typeid(A)` from another module
// even when `A` is the same, non-hidden-visibility type (e.g. from a common include).  Under
// libstdc++, this doesn't happen: equality and the type_index hash are based on the type name,
// which works.  If not under a known-good stl, provide our own name-based hash and equality
// functions that use the type name.
#if defined(__GLIBCXX__)

inline bool same_type(const std::type_info &lhs, const std::type_info &rhs) { return lhs == rhs; }
using type_hash = std::hash<std::type_index>;
using type_equal_to = std::equal_to<std::type_index>;

#else

inline bool same_type(const std::type_info &lhs, const std::type_info &rhs) {
    return lhs.name() == rhs.name() || std::strcmp(lhs.name(), rhs.name()) == 0;
}

struct type_hash {
    size_t operator()(const std::type_index &t) const {
        size_t hash = 5381;
        const char *ptr = t.name();
        while (auto c = static_cast<unsigned char>(*ptr++)) {
            hash = (hash * 33) ^ c;
        }
        return hash;
    }
};

struct type_equal_to {
    bool operator()(const std::type_index &lhs, const std::type_index &rhs) const {
        return lhs.name() == rhs.name() || std::strcmp(lhs.name(), rhs.name()) == 0;
    }
};

#endif

template <typename value_type>
using type_map = std::unordered_map<std::type_index, value_type, type_hash, type_equal_to>;

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
