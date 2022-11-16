// Copyright (c) 2022 The pybind Community.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include "../pytypes.h"
#include "common.h"

#include <typeindex>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

class native_enum_data {
public:
    native_enum_data(const char *enum_name,
                     const std::type_index &enum_type_index,
                     bool use_int_enum)
        : enum_name{enum_name}, enum_type_index{enum_type_index}, use_int_enum{use_int_enum} {}

    std::string enum_name;
    std::type_index enum_type_index;
    bool use_int_enum;
    bool export_values_flag = false;
    list members;
    list docs;

    native_enum_data(const native_enum_data &) = delete;
    native_enum_data &operator=(const native_enum_data &) = delete;

    // This is a separate public function only to enable easy unit testing.
    std::string was_not_added_error_message() const {
        return "`native_enum` was not added to any module."
               " Use e.g. `m += native_enum<...>(\""
               + enum_name + "\")` to fix.";
    }

#if defined(NDEBUG)
    void set_was_added_to_module() const {};
#else
    void set_was_added_to_module() const { was_added_to_module = true; }

private:
    mutable bool was_added_to_module = false;

public:
    // This dtor cannot easily be unit tested because it terminates the process.
    ~native_enum_data() {
        if (!was_added_to_module) {
            pybind11_fail(was_not_added_error_message());
        }
    }
#endif
};

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
