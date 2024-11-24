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

#include <algorithm>

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

template <typename T>
class Type : public type {
    using type::type;
};

template <typename... Types>
class Union : public object {
    PYBIND11_OBJECT_DEFAULT(Union, object, PyObject_Type)
    using object::object;
};

template <typename T>
class Optional : public object {
    PYBIND11_OBJECT_DEFAULT(Optional, object, PyObject_Type)
    using object::object;
};

template <typename T>
class TypeGuard : public bool_ {
    using bool_::bool_;
};

template <typename T>
class TypeIs : public bool_ {
    using bool_::bool_;
};

class NoReturn : public none {
    using none::none;
};

class Never : public none {
    using none::none;
};

#if defined(__cpp_nontype_template_args) && __cpp_nontype_template_args >= 201911L
#    define PYBIND11_TYPING_H_HAS_STRING_LITERAL
template <size_t N>
struct StringLiteral {
    constexpr StringLiteral(const char (&str)[N]) { std::copy_n(str, N, name); }
    char name[N];
};

template <StringLiteral... StrLits>
class Literal : public object {
    PYBIND11_OBJECT_DEFAULT(Literal, object, PyObject_Type)
};

// Example syntax for creating a TypeVar.
// typedef typing::TypeVar<"T"> TypeVarT;
template <StringLiteral>
class TypeVar : public object {
    PYBIND11_OBJECT_DEFAULT(TypeVar, object, PyObject_Type)
    using object::object;
};
#endif

PYBIND11_NAMESPACE_END(typing)

PYBIND11_NAMESPACE_BEGIN(detail)

template <typename... Types>
struct handle_type_name<typing::Tuple<Types...>> {
    template <typename... Ds>
    static constexpr auto wrap_name(const Ds &...names) {
        return const_name("tuple[") + ::pybind11::detail::concat(names...) + const_name("]");
    }
    static constexpr auto name = wrap_name(make_caster<Types>::name...);
    static constexpr auto arg_name = wrap_name(as_arg_type<make_caster<Types>>::name...);
    static constexpr auto return_name = wrap_name(as_return_type<make_caster<Types>>::name...);
};

template <>
struct handle_type_name<typing::Tuple<>> {
    // PEP 484 specifies this syntax for an empty tuple
    static constexpr auto name = const_name("tuple[()]");
};

template <typename T>
struct handle_type_name<typing::Tuple<T, ellipsis>> {
    // PEP 484 specifies this syntax for a variable-length tuple
    template <typename D>
    static constexpr auto wrap_name(const D &name) {
        return const_name("tuple[") + name + const_name(", ...]");
    }
    static constexpr auto name = wrap_name(make_caster<T>::name);
    static constexpr auto arg_name = wrap_name(as_arg_type<make_caster<T>>::name);
    static constexpr auto return_name = wrap_name(as_return_type<make_caster<T>>::name);
};

template <typename K, typename V>
struct handle_type_name<typing::Dict<K, V>> {
    template <typename Kd, typename Vd>
    static constexpr auto wrap_name(const Kd &key_name, const Vd &value_name) {
        return const_name("dict[") + key_name + const_name(", ") + value_name + const_name("]");
    }
    static constexpr auto name = wrap_name(make_caster<K>::name, make_caster<V>::name);
    static constexpr auto arg_name
        = wrap_name(as_arg_type<make_caster<K>>::name, as_arg_type<make_caster<V>>::name);
    static constexpr auto return_name
        = wrap_name(as_return_type<make_caster<K>>::name, as_return_type<make_caster<V>>::name);
};

template <typename T>
struct handle_type_name<typing::List<T>> {
    template <typename D>
    static constexpr auto wrap_name(const D &name) {
        return const_name("list[") + name + const_name("]");
    }
    static constexpr auto name = wrap_name(make_caster<T>::name);
    static constexpr auto arg_name = wrap_name(as_arg_type<make_caster<T>>::name);
    static constexpr auto return_name = wrap_name(as_return_type<make_caster<T>>::name);
};

template <typename T>
struct handle_type_name<typing::Set<T>> {
    template <typename D>
    static constexpr auto wrap_name(const D &name) {
        return const_name("set[") + name + const_name("]");
    }
    static constexpr auto name = wrap_name(make_caster<T>::name);
    static constexpr auto arg_name = wrap_name(as_arg_type<make_caster<T>>::name);
    static constexpr auto return_name = wrap_name(as_return_type<make_caster<T>>::name);
};

template <typename T>
struct handle_type_name<typing::Iterable<T>> {
    template <typename D>
    static constexpr auto wrap_name(const D &name) {
        return const_name("Iterable[") + name + const_name("]");
    }
    static constexpr auto name = wrap_name(make_caster<T>::name);
    static constexpr auto arg_name = wrap_name(as_arg_type<make_caster<T>>::name);
    static constexpr auto return_name = wrap_name(as_return_type<make_caster<T>>::name);
};

template <typename T>
struct handle_type_name<typing::Iterator<T>> {
    template <typename D>
    static constexpr auto wrap_name(const D &name) {
        return const_name("Iterator[") + name + const_name("]");
    }
    static constexpr auto name = wrap_name(make_caster<T>::name);
    static constexpr auto arg_name = wrap_name(as_arg_type<make_caster<T>>::name);
    static constexpr auto return_name = wrap_name(as_return_type<make_caster<T>>::name);
};

template <typename Return, typename... Args>
struct handle_type_name<typing::Callable<Return(Args...)>> {
    using retval_type = conditional_t<std::is_same<Return, void>::value, void_type, Return>;
    static constexpr auto name
        = const_name("Callable[[") + ::pybind11::detail::concat(make_caster<Args>::name...)
          + const_name("], ") + make_caster<retval_type>::name + const_name("]");
};

template <typename Return>
struct handle_type_name<typing::Callable<Return(ellipsis)>> {
    // PEP 484 specifies this syntax for defining only return types of callables
    using retval_type = conditional_t<std::is_same<Return, void>::value, void_type, Return>;
    static constexpr auto name
        = const_name("Callable[..., ") + make_caster<retval_type>::name + const_name("]");
};

template <typename T>
struct handle_type_name<typing::Type<T>> {
    static constexpr auto name = const_name("type[") + make_caster<T>::name + const_name("]");
};

template <typename... Types>
struct handle_type_name<typing::Union<Types...>> {
    template <typename... Ds>
    static constexpr auto wrap_name(const Ds &...names) {
        return const_name("Union[") + ::pybind11::detail::concat(names...) + const_name("]");
    }
    static constexpr auto name = wrap_name(make_caster<Types>::name...);
    static constexpr auto arg_name = wrap_name(as_arg_type<make_caster<Types>>::name...);
    static constexpr auto return_name = wrap_name(as_return_type<make_caster<Types>>::name...);
};

template <typename T>
struct handle_type_name<typing::Optional<T>> {
    template <typename D>
    static constexpr auto wrap_name(const D &name) {
        return const_name("Optional[") + name + const_name("]");
    }
    static constexpr auto name = wrap_name(make_caster<T>::name);
    static constexpr auto arg_name = wrap_name(as_arg_type<make_caster<T>>::name);
    static constexpr auto return_name = wrap_name(as_return_type<make_caster<T>>::name);
};

template <typename T>
struct handle_type_name<typing::TypeGuard<T>> {
    template <typename D>
    static constexpr auto wrap_name(const D &name) {
        return const_name("TypeGuard[") + name + const_name("]");
    }
    static constexpr auto name = wrap_name(make_caster<T>::name);
    static constexpr auto arg_name = wrap_name(as_arg_type<make_caster<T>>::name);
    static constexpr auto return_name = wrap_name(as_return_type<make_caster<T>>::name);
};

template <typename T>
struct handle_type_name<typing::TypeIs<T>> {
    template <typename D>
    static constexpr auto wrap_name(const D &name) {
        return const_name("TypeIs[") + name + const_name("]");
    }
    static constexpr auto name = wrap_name(make_caster<T>::name);
    static constexpr auto arg_name = wrap_name(as_arg_type<make_caster<T>>::name);
    static constexpr auto return_name = wrap_name(as_return_type<make_caster<T>>::name);
};

template <>
struct handle_type_name<typing::NoReturn> {
    static constexpr auto name = const_name("NoReturn");
};

template <>
struct handle_type_name<typing::Never> {
    static constexpr auto name = const_name("Never");
};

#if defined(PYBIND11_TYPING_H_HAS_STRING_LITERAL)
template <typing::StringLiteral... Literals>
struct handle_type_name<typing::Literal<Literals...>> {
    static constexpr auto name = const_name("Literal[")
                                 + pybind11::detail::concat(const_name(Literals.name)...)
                                 + const_name("]");
};
template <typing::StringLiteral StrLit>
struct handle_type_name<typing::TypeVar<StrLit>> {
    static constexpr auto name = const_name(StrLit.name);
};
#endif

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
