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
    native_enum &export_values() {
        m_export_values = true;
        return *this;
    }

    /// Add an enumeration entry
    native_enum &value(char const *name, Type value, const char *doc = nullptr) {
        m_members.append(make_tuple(name, static_cast<Underlying>(value)));
        if (doc) {
            m_docs.append(make_tuple(name, doc));
        }
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
        auto py_enum_type = enum_module.attr(use_int_enum ? "IntEnum" : "Enum");
        auto py_enum = py_enum_type(m_name, m_members);
        py_enum.attr("__module__") = m_scope;
        m_scope.attr(m_name) = py_enum;
        if (m_export_values) {
            for (auto member : m_members) {
                auto member_name = member[int_(0)];
                m_scope.attr(member_name) = py_enum[member_name];
            }
        }
        for (auto doc : m_docs) {
            py_enum[doc[int_(0)]].attr("__doc__") = doc[int_(1)];
        }
        // Intentionally leak Python reference.
        detail::get_internals().native_enum_types[std::type_index(typeid(Type))]
            = py_enum.release().ptr();
    }

private:
    object m_scope;
    str m_name;
    bool m_export_values = false;
    list m_members;
    list m_docs;
};

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
