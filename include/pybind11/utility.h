/*
    pybind11/utility.h: Support for STL utility types (variant, optional)

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "pybind11.h"

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#endif

#ifdef __has_include
// std::optional (but including it in c++14 mode isn't allowed)
#  if defined(PYBIND11_CPP17) && __has_include(<optional>)
#    include <optional>
#    define PYBIND11_HAS_OPTIONAL 1
#  endif
// std::experimental::optional (but not allowed in c++11 mode)
#  if defined(PYBIND11_CPP14) && __has_include(<experimental/optional>)
#    include <experimental/optional>
#    define PYBIND11_HAS_EXP_OPTIONAL 1
#  endif
// std::variant
#  if defined(PYBIND11_CPP17) && __has_include(<variant>)
#    include <variant>
#    define PYBIND11_HAS_VARIANT 1
#  endif
#elif defined(_MSC_VER) && defined(PYBIND11_CPP17)
#  include <optional>
#  include <variant>
#  define PYBIND11_HAS_OPTIONAL 1
#  define PYBIND11_HAS_VARIANT 1
#endif

NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
NAMESPACE_BEGIN(detail)

// This type caster is intended to be used for std::optional and std::experimental::optional
template<typename T> struct optional_caster {
    using value_conv = make_caster<typename T::value_type>;

    template <typename T_>
    static handle cast(T_ &&src, return_value_policy policy, handle parent) {
        if (!src)
            return none().inc_ref();
        return value_conv::cast(*std::forward<T_>(src), policy, parent);
    }

    bool load(handle src, bool convert) {
        if (!src) {
            return false;
        } else if (src.is_none()) {
            return true;  // default-constructed value is already empty
        }
        value_conv inner_caster;
        if (!inner_caster.load(src, convert))
            return false;

        value.emplace(cast_op<typename T::value_type &&>(std::move(inner_caster)));
        return true;
    }

    PYBIND11_TYPE_CASTER(T, _("Optional[") + value_conv::name + _("]"));
};

#if PYBIND11_HAS_OPTIONAL
template<typename T> struct type_caster<std::optional<T>>
    : public optional_caster<std::optional<T>> {};

template<> struct type_caster<std::nullopt_t>
    : public void_caster<std::nullopt_t> {};
#endif

#if PYBIND11_HAS_EXP_OPTIONAL
template<typename T> struct type_caster<std::experimental::optional<T>>
    : public optional_caster<std::experimental::optional<T>> {};

template<> struct type_caster<std::experimental::nullopt_t>
    : public void_caster<std::experimental::nullopt_t> {};
#endif

/// Visit a variant and cast any found type to Python
struct variant_caster_visitor {
    return_value_policy policy;
    handle parent;

    using result_type = handle; // required by boost::variant in C++11

    template <typename T>
    result_type operator()(T &&src) const {
        return make_caster<T>::cast(std::forward<T>(src), policy, parent);
    }
};

/// Helper class which abstracts away variant's `visit` function. `std::variant` and similar
/// `namespace::variant` types which provide a `namespace::visit()` function are handled here
/// automatically using argument-dependent lookup. Users can provide specializations for other
/// variant-like classes, e.g. `boost::variant` and `boost::apply_visitor`.
template <template<typename...> class Variant>
struct visit_helper {
    template <typename... Args>
    static auto call(Args &&...args) -> decltype(visit(std::forward<Args>(args)...)) {
        return visit(std::forward<Args>(args)...);
    }
};

/// Generic variant caster
template <typename Variant> struct variant_caster;

template <template<typename...> class V, typename... Ts>
struct variant_caster<V<Ts...>> {
    static_assert(sizeof...(Ts) > 0, "Variant must consist of at least one alternative.");

    template <typename U, typename... Us>
    bool load_alternative(handle src, bool convert, type_list<U, Us...>) {
        auto caster = make_caster<U>();
        if (caster.load(src, convert)) {
            value = cast_op<U>(caster);
            return true;
        }
        return load_alternative(src, convert, type_list<Us...>{});
    }

    bool load_alternative(handle, bool, type_list<>) { return false; }

    bool load(handle src, bool convert) {
        // Do a first pass without conversions to improve constructor resolution.
        // E.g. `py::int_(1).cast<variant<double, int>>()` needs to fill the `int`
        // slot of the variant. Without two-pass loading `double` would be filled
        // because it appears first and a conversion is possible.
        if (convert && load_alternative(src, false, type_list<Ts...>{}))
            return true;
        return load_alternative(src, convert, type_list<Ts...>{});
    }

    template <typename Variant>
    static handle cast(Variant &&src, return_value_policy policy, handle parent) {
        return visit_helper<V>::call(variant_caster_visitor{policy, parent},
                                     std::forward<Variant>(src));
    }

    using Type = V<Ts...>;
    PYBIND11_TYPE_CASTER(Type, _("Union[") + detail::concat(make_caster<Ts>::name...) + _("]"));
};

#if PYBIND11_HAS_VARIANT
template <typename... Ts>
struct type_caster<std::variant<Ts...>> : variant_caster<std::variant<Ts...>> { };
#endif

NAMESPACE_END(detail)

NAMESPACE_END(PYBIND11_NAMESPACE)

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
