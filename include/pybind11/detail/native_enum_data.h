// Copyright (c) 2022 The pybind Community.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include "../pytypes.h"
#include "common.h"

#include <typeindex>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

struct native_enum_data {
    native_enum_data(const char *name, const std::type_index &enum_type_index, bool use_int_enum)
        : name{name}, enum_type_index{enum_type_index}, use_int_enum{use_int_enum} {}

    str name;
    std::type_index enum_type_index;
    bool use_int_enum;
    bool export_values_flag = false;
    list members;
    list docs;
};

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
