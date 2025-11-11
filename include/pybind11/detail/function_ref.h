/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//===- llvm/ADT/STLFunctionalExtras.h - Extras for <functional> -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a header-only class template that provides functionality
// similar to std::function but with non-owning semantics. It is a template-only
// implementation that requires no additional library linking.
//
//===----------------------------------------------------------------------===//

/// An efficient, type-erasing, non-owning reference to a callable. This is
/// intended for use as the type of a function parameter that is not used
/// after the function in question returns.
///
/// This class does not own the callable, so it is not in general safe to store
/// a FunctionRef.

// pybind11: modified again from executorch::runtime::FunctionRef
// - renamed back to function_ref
// - use pybind11 enable_if_t, remove_cvref_t, and remove_reference_t
// - lint suppressions

// torch::executor: modified from llvm::function_ref
// - renamed to FunctionRef
// - removed LLVM_GSL_POINTER and LLVM_LIFETIME_BOUND macro uses
// - use namespaced internal::remove_cvref_t

#pragma once

#include <pybind11/detail/common.h>

#include <cstdint>
#include <type_traits>
#include <utility>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

//===----------------------------------------------------------------------===//
//     Features from C++20
//===----------------------------------------------------------------------===//

template <typename Fn>
class function_ref;

template <typename Ret, typename... Params>
class function_ref<Ret(Params...)> {
    Ret (*callback)(intptr_t callable, Params... params) = nullptr;
    intptr_t callable;

    template <typename Callable>
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    static Ret callback_fn(intptr_t callable, Params... params) {
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        return (*reinterpret_cast<Callable *>(callable))(std::forward<Params>(params)...);
    }

public:
    function_ref() = default;
    // NOLINTNEXTLINE(google-explicit-constructor)
    function_ref(std::nullptr_t) {}

    template <typename Callable>
    // NOLINTNEXTLINE(google-explicit-constructor)
    function_ref(
        Callable &&callable,
        // This is not the copy-constructor.
        enable_if_t<!std::is_same<remove_cvref_t<Callable>, function_ref>::value> * = nullptr,
        // Functor must be callable and return a suitable type.
        enable_if_t<
            std::is_void<Ret>::value
            || std::is_convertible<decltype(std::declval<Callable>()(std::declval<Params>()...)),
                                   Ret>::value> * = nullptr)
        : callback(callback_fn<remove_reference_t<Callable>>),
          callable(reinterpret_cast<intptr_t>(&callable)) {}

    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    Ret operator()(Params... params) const {
        return callback(callable, std::forward<Params>(params)...);
    }

    explicit operator bool() const { return callback; }

    bool operator==(const function_ref<Ret(Params...)> &Other) const {
        return callable == Other.callable;
    }
};
PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
