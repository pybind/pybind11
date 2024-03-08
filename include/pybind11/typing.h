/*
    pybind11/typing.h: Convenience wrapper classes for basic Python types
    with more explicit annotations.

    Copyright (c) 2023 Dustin Spicuzza <dustin@virtualroadside.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "detail/common.h"
#include "cast.h"
#include "pytypes.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(typing)

/*
    The following types can be used to direct pybind11-generated docstrings
    to have have more explicit types (e.g., `list[str]` instead of `list`).
    Just use these in place of existing types.

    There is no additional enforcement of types at runtime.
*/

template <typename... Types>
class Tuple : public tuple {
    using tuple::tuple;
};

template <typename K, typename V>
class Dict : public dict {
    using dict::dict;
};

template <typename T>
class List : public list {
    using list::list;
};

template <typename T>
class Set : public set {
    using set::set;
};

template <typename T>
class Iterable : public iterable {
    using iterable::iterable;
};

template <typename T>
class Iterator : public iterator {
    using iterator::iterator;
};

template <typename Signature>
class Callable;

template <typename Return, typename... Args>
class Callable<Return(Args...)> : public function {
    using function::function;
};

PYBIND11_NAMESPACE_END(typing)

PYBIND11_NAMESPACE_BEGIN(detail)

template <typename... Types>
struct handle_type_name<typing::Tuple<Types...>> {
    static constexpr auto name
        = const_name("tuple[") + concat(make_caster<Types>::name...) + const_name("]");
};

template <>
struct handle_type_name<typing::Tuple<>> {
    // PEP 484 specifies this syntax for an empty tuple
    static constexpr auto name = const_name("tuple[()]");
};

template <typename T>
struct handle_type_name<typing::Tuple<T, ellipsis>> {
    // PEP 484 specifies this syntax for a variable-length tuple
    static constexpr auto name
        = const_name("tuple[") + make_caster<T>::name + const_name(", ...]");
};

template <typename K, typename V>
struct handle_type_name<typing::Dict<K, V>> {
    static constexpr auto name = const_name("dict[") + make_caster<K>::name + const_name(", ")
                                 + make_caster<V>::name + const_name("]");
};

template <typename T>
struct handle_type_name<typing::List<T>> {
    static constexpr auto name = const_name("list[") + make_caster<T>::name + const_name("]");
};

template <typename T>
struct handle_type_name<typing::Set<T>> {
    static constexpr auto name = const_name("set[") + make_caster<T>::name + const_name("]");
};

template <typename T>
struct handle_type_name<typing::Iterable<T>> {
    static constexpr auto name = const_name("Iterable[") + make_caster<T>::name + const_name("]");
};

template <typename T>
struct handle_type_name<typing::Iterator<T>> {
    static constexpr auto name = const_name("Iterator[") + make_caster<T>::name + const_name("]");
};

template <typename Return, typename... Args>
struct handle_type_name<typing::Callable<Return(Args...)>> {
    using retval_type = conditional_t<std::is_same<Return, void>::value, void_type, Return>;
    static constexpr auto name = const_name("Callable[[") + concat(make_caster<Args>::name...)
                                 + const_name("], ") + make_caster<retval_type>::name
                                 + const_name("]");
};

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
