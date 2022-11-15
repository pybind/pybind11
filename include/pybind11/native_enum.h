// Copyright (c) 2022 The pybind Community.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include "pybind11.h"

#include <type_traits>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

/// Conversions between Python's native (stdlib) enum types and C++ enums.
template <typename Type>
class native_enum {
public:
    using Underlying = typename std::underlying_type<Type>::type;

    native_enum(const object &scope, const char *name) : m_scope(scope), m_name(name) {}

    /// Export enumeration entries into the parent scope
    native_enum &export_values() { return *this; }

    /// Add an enumeration entry
    native_enum &value(char const *name, Type value, const char *doc = nullptr) {
        if (doc) {
            // IGNORED.
        }
        m_members.append(make_tuple(name, static_cast<Underlying>(value)));
        return *this;
    }

    native_enum(const native_enum &) = delete;
    native_enum &operator=(const native_enum &) = delete;

    ~native_enum() {
        // Any exception here will terminate the process.
        auto enum_module = module_::import("enum");
        constexpr bool use_int_enum = std::numeric_limits<Underlying>::is_integer
                                      && !std::is_same<Underlying, bool>::value
                                      && !detail::is_std_char_type<Underlying>::value;
        auto int_enum = enum_module.attr(use_int_enum ? "IntEnum" : "Enum");
        auto int_enum_color = int_enum(m_name, m_members);
        int_enum_color.attr("__module__") = m_scope;
        m_scope.attr(m_name) = int_enum_color;
        // Intentionally leak Python reference.
        detail::get_internals().native_enum_types[std::type_index(typeid(Type))]
            = int_enum_color.release().ptr();
    }

private:
    object m_scope;
    str m_name;
    list m_members;
};

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
