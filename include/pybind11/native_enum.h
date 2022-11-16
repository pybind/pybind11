// Copyright (c) 2022 The pybind Community.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include "detail/common.h"
#include "detail/native_enum_data.h"
#include "cast.h"

#include <limits>
#include <type_traits>
#include <typeindex>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

/// Conversions between Python's native (stdlib) enum types and C++ enums.
template <typename Type>
class native_enum : public detail::native_enum_data {
public:
    using Underlying = typename std::underlying_type<Type>::type;

    explicit native_enum(const char *name)
        : detail::native_enum_data(name,
                                   std::type_index(typeid(Type)),
                                   std::numeric_limits<Underlying>::is_integer
                                       && !std::is_same<Underlying, bool>::value
                                       && !detail::is_std_char_type<Underlying>::value) {}

    /// Export enumeration entries into the parent scope
    native_enum &export_values() {
        export_values_flag = true;
        return *this;
    }

    /// Add an enumeration entry
    native_enum &value(char const *name, Type value, const char *doc = nullptr) {
        members.append(make_tuple(name, static_cast<Underlying>(value)));
        if (doc) {
            docs.append(make_tuple(name, doc));
        }
        return *this;
    }

    native_enum(const native_enum &) = delete;
    native_enum &operator=(const native_enum &) = delete;
};

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
