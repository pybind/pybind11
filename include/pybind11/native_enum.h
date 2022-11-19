// Copyright (c) 2022 The pybind Community.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include "detail/common.h"
#include "detail/native_enum_data.h"
#include "detail/type_caster_base.h"
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
                                       && !detail::is_std_char_type<Underlying>::value) {
        if (detail::get_local_type_info(typeid(Type)) != nullptr
            || detail::get_global_type_info(typeid(Type)) != nullptr) {
            pybind11_fail(
                "pybind11::native_enum<...>(\"" + enum_name_encoded
                + "\") is already registered as a `pybind11::enum_` or `pybind11::class_`!");
        }
        if (cross_extension_shared_states::native_enum_type_map::get().count(enum_type_index)) {
            pybind11_fail("pybind11::native_enum<...>(\"" + enum_name_encoded
                          + "\") is already registered!");
        }
        arm_correct_use_check();
    }

    /// Export enumeration entries into the parent scope
    native_enum &export_values() {
        export_values_flag = true;
        return *this;
    }

    /// Add an enumeration entry
    native_enum &value(char const *name, Type value, const char *doc = nullptr) {
        disarm_correct_use_check();
        members.append(make_tuple(name, static_cast<Underlying>(value)));
        if (doc) {
            docs.append(make_tuple(name, doc));
        }
        arm_correct_use_check();
        return *this;
    }

    native_enum(const native_enum &) = delete;
    native_enum &operator=(const native_enum &) = delete;
};

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
