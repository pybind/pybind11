// Copyright (c) 2024 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include "pybind11.h"

#define PYBIND11_TYPE_CASTER_BASE_HOLDER(...)
#define PYBIND11_SMART_HOLDER_TYPE_CASTERS(...)
#define PYBIND11_SH_AVL(...) // "Smart_Holder if AVaiLable"
#define PYBIND11_SH_DEF(...) // "Smart_Holder if DEFault"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

template <typename type_, typename... options>
class classh : public class_<type_, options...> {
public:
    using class_<type_, options...>::class_;
};

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
