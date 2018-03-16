/*
    pybind11/detail/inference.h -- Simple inference for generic functions

    Copyright (c) 2018 Eric Cousineau <eric.cousineau@tri.global>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"

NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
NAMESPACE_BEGIN(detail)

// SFINAE for functors.
// N.B. This *only* distinguished between function / method pointers and
// lambda objects. It does *not* distinguish among other types.
template <typename Func, typename T = void>
using enable_if_lambda_t = enable_if_t<!std::is_function<intrinsic_t<Func>>::value, T>;

template <size_t N, size_t K, typename T, typename ... Ts>
struct type_at_impl {
  using type = typename type_at_impl<N, K + 1, Ts...>::type;
};

template <size_t N, typename T, typename ... Ts>
struct type_at_impl<N, N, T, Ts...> {
  using type = T;
};

// Convenient mechanism for passing sets of arguments.
template <typename ... Ts>
struct type_pack {
    static constexpr int size = sizeof...(Ts);

    template <template <typename...> class Tpl>
    using bind = Tpl<Ts...>;

    template <size_t N>
    struct type_at_internal {
      static_assert(N < size, "Invalid type index");
      using type = typename type_at_impl<N, 0, Ts...>::type;
    };

    template <size_t N>
    using type_at = typename type_at_internal<N>::type;
};

template <typename... A, typename... B>
auto type_pack_concat(type_pack<A...> = {}, type_pack<B...> = {}) {
  return type_pack<A..., B...>{};
}

template <template <typename> class Apply, typename... T>
auto type_pack_apply(type_pack<T...> = {}) {
  return type_pack<Apply<T>...>{};
}

struct function_inference {
    // Collects both a functor object and its signature for ease of inference.
    template <typename Func, typename ReturnT, typename ... ArgsT>
    struct inferred_info {
      // TODO(eric.cousineau): Ensure that this permits copy elision when combined
      // with `std::forward<Func>(func)`, while still behaving well with primitive
      // types.
      std::decay_t<Func> func;

      using Return = ReturnT;
      using Args = type_pack<ArgsT...>;
    };

    // Factory method for `inferred_info<>`, to be used by `run`.
    template <typename Return, typename ... Args, typename Func>
    static auto make_inferred_info(Func&& func, Return (*infer)(Args...) = nullptr) {
      (void)infer;
      return inferred_info<Func, Return, Args...>{std::forward<Func>(func)};
    }

    // Infers `inferred_info<>` from a function pointer.
    template <typename Return, typename ... Args>
    static auto run(Return (*func)(Args...)) {
      return make_inferred_info<Return, Args...>(func);
    }

    // Infers `inferred_info<>` from a mutable method pointer.
    template <typename Return, typename Class, typename ... Args>
    static auto run(Return (Class::*method)(Args...)) {
      auto func = [method](Class& self, Args... args) {
        return (self.*method)(std::forward<Args>(args)...);
      };
      return make_inferred_info<Return, Class&, Args...>(func);
    }

    // Infers `inferred_info<>` from a const method pointer.
    template <typename Return, typename Class, typename ... Args>
    static auto run(Return (Class::*method)(Args...) const) {
      auto func = [method](const Class& self, Args... args) {
        return (self.*method)(std::forward<Args>(args)...);
      };
      return make_inferred_info<Return, const Class&, Args...>(func);
    }

    // Helpers for general functor objects.
    struct infer_helper {
      // Removes class from mutable method pointer for inferring signature
      // of functor.
      template <typename Class, typename Return, typename ... Args>
      static auto remove_class_from_ptr(Return (Class::*)(Args...)) {
        using Ptr = Return (*)(Args...);
        return Ptr{};
      }

      // Removes class from const method pointer for inferring signature of functor.
      template <typename Class, typename Return, typename ... Args>
      static auto remove_class_from_ptr(Return (Class::*)(Args...) const) {
        using Ptr = Return (*)(Args...);
        return Ptr{};
      }

      // Infers funtion pointer from functor.
      // @pre `Func` must have only *one* overload of `operator()`.
      template <typename Func>
      static auto infer_function_ptr() {
        return remove_class_from_ptr(&Func::operator());
      }
    };

    // Infers `inferred_info<>` from a generic functor.
    template <typename Func, typename = detail::enable_if_lambda_t<Func>>
    static auto run(Func&& func) {
      return make_inferred_info(
          std::forward<Func>(func),
          infer_helper::infer_function_ptr<std::decay_t<Func>>());
    }
};

NAMESPACE_END(detail)

NAMESPACE_END(PYBIND11_NAMESPACE)
