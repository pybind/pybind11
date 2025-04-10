// Copyright (c) 2020-2024 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include "pybind11/detail/struct_smart_holder.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(memory)
PYBIND11_NAMESPACE_BEGIN(smart_holder_poc) // Proof-of-Concept implementations.

struct PrivateESFT : private std::enable_shared_from_this<PrivateESFT> {};
static_assert(!type_has_shared_from_this(static_cast<PrivateESFT *>(nullptr)),
              "should detect inaccessible base");

template <typename T>
T &as_lvalue_ref(const smart_holder &hld) {
    static const char *context = "as_lvalue_ref";
    hld.ensure_is_populated(context);
    hld.ensure_has_pointee(context);
    return *hld.as_raw_ptr_unowned<T>();
}

template <typename T>
T &&as_rvalue_ref(const smart_holder &hld) {
    static const char *context = "as_rvalue_ref";
    hld.ensure_is_populated(context);
    hld.ensure_has_pointee(context);
    return std::move(*hld.as_raw_ptr_unowned<T>());
}

template <typename T>
T *as_raw_ptr_release_ownership(smart_holder &hld,
                                const char *context = "as_raw_ptr_release_ownership") {
    hld.ensure_can_release_ownership(context);
    T *raw_ptr = hld.as_raw_ptr_unowned<T>();
    hld.release_ownership();
    return raw_ptr;
}

template <typename T, typename D = std::default_delete<T>>
std::unique_ptr<T, D> as_unique_ptr(smart_holder &hld) {
    static const char *context = "as_unique_ptr";
    hld.ensure_compatible_rtti_uqp_del<T, D>(context);
    hld.ensure_use_count_1(context);
    T *raw_ptr = hld.as_raw_ptr_unowned<T>();
    hld.release_ownership();
    // KNOWN DEFECT (see PR #4850): Does not copy the deleter.
    return std::unique_ptr<T, D>(raw_ptr);
}

PYBIND11_NAMESPACE_END(smart_holder_poc)
PYBIND11_NAMESPACE_END(memory)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
