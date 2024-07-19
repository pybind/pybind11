// Copyright (c) 2024 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include "common.h"
#include "internals.h"

#include <type_traits>

#ifdef PYBIND11_HAVE_INTERNALS_WITH_SMART_HOLDER_SUPPORT
#    include "smart_holder_poc.h"
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

#ifdef PYBIND11_HAVE_INTERNALS_WITH_SMART_HOLDER_SUPPORT
using pybindit::memory::smart_holder;
#endif

PYBIND11_NAMESPACE_BEGIN(detail)

#ifdef PYBIND11_HAVE_INTERNALS_WITH_SMART_HOLDER_SUPPORT
template <typename H>
using is_smart_holder = std::is_same<H, smart_holder>;
#else
template <typename>
struct is_smart_holder : std::false_type {};
#endif

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
