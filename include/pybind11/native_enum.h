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
    // Scalar is the integer representation of underlying type
    using Scalar = detail::conditional_t<detail::any_of<detail::is_std_char_type<Underlying>,
                                                        std::is_same<Underlying, bool>>::value,
                                         detail::equivalent_integer_t<Underlying>,
                                         Underlying>;

    template <typename... Extra>
    native_enum(const handle &scope, const char *name, const Extra &.../*extra*/)
        : m_scope(reinterpret_borrow<object>(scope)), m_name(name) {
        constexpr bool is_arithmetic = detail::any_of<std::is_same<arithmetic, Extra>...>::value;
        constexpr bool is_convertible = std::is_convertible<Type, Underlying>::value;
        if (is_arithmetic || is_convertible) {
            // IGNORED.
        }
    }

    /// Export enumeration entries into the parent scope
    native_enum &export_values() { return *this; }

    /// Add an enumeration entry
    native_enum &value(char const *name, Type value, const char *doc = nullptr) {
        if (doc) {
            // IGNORED.
        }
        m_members.append(make_tuple(name, static_cast<Scalar>(value)));
        return *this;
    }

    native_enum(const native_enum &) = delete;
    native_enum &operator=(const native_enum &) = delete;

    ~native_enum() {
        // Any exception here will terminate the process.
        auto enum_module = module_::import("enum");
        auto int_enum = enum_module.attr("IntEnum");
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
