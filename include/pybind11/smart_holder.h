// Copyright (c) 2021 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include "pybind11.h"
#include "detail/common.h"
#include "detail/smart_holder_type_casters.h"

#undef PYBIND11_SH_AVL // Undoing #define in pybind11.h

#define PYBIND11_SH_AVL(...) ::pybind11::smart_holder // "Smart_Holder if AVaiLable"
// ---- std::shared_ptr(...) -- same length by design, to not disturb the indentation
// of existing code.

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

// Supports easier switching between py::class_<T> and py::class_<T, py::smart_holder>:
// users can simply replace the `_` in `class_` with `h` or vice versa.
// Note though that the PYBIND11_SMART_HOLDER_TYPE_CASTERS(T) macro also needs to be
// added (for `classh`) or commented out (when falling back to `class_`).
template <typename type_, typename... options>
class classh : public class_<type_, smart_holder, options...> {
public:
    using class_<type_, smart_holder, options...>::class_;
};

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
