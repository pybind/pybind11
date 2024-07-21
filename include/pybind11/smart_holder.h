// Copyright (c) 2021-2024 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include "pybind11.h"

// Legacy macros introduced with smart_holder_type_casters implementation in 2021.
// Deprecated.
#define PYBIND11_TYPE_CASTER_BASE_HOLDER(...)
#define PYBIND11_SMART_HOLDER_TYPE_CASTERS(...)
#define PYBIND11_SH_AVL(...) // "Smart_Holder if AVaiLable"
#define PYBIND11_SH_DEF(...) // "Smart_Holder if DEFault"
