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

#if defined(__cpp_nontype_template_parameter_class)
template <size_t N>
struct StringLiteral {
    constexpr StringLiteral(const char (&str)[N]) { std::copy_n(str, N, value); }

    char value[N];
};

template <StringLiteral lit>
class TypeVar : public object {
    PYBIND11_OBJECT_DEFAULT(TypeVar, object, PyObject_Type)
    using object::object;
};


// Does not currently support Literals of byte strings, unicode strings, and Enum values.
// Also due to how C++ implemented constant template Literal[1, 2] does not equal Literal[2, 1]
// template <StringLiteral... lit>
// class Literal : public str {
//     using str::str;
// };

template <typename inputT, typename T>
class Literal : public T {
    // if std::is_same<T, object>{
    //     PYBIND11_OBJECT_DEFAULT(TypeVar, object, PyObject_Type);
    // }
    using T::T;
};

template<StringLiteral... literal>
typedef Literal<StringLiteral, str> LiteralStr;

typedef LiteralStr<"1", "2"> LiteralStrOneTwo;


// typedef Literal<StringLiteral, str, "3", "4"> LiteralStrThreeFour;

// template <int... intLit>
// class Literal : public py::int_ {
//     using int_::int_;
// };

// template <bool... boolLit>
// class Literal : public bool_ {
//     using bool_::bool_;
// };

// template <py::none>
// class Literal : public none {
//     using none::none;
// };

// template <any... anyLit>
// class Literal : public py::object {
//     PYBIND11_OBJECT_DEFAULT(TypeVar, object, PyObject_Type)
//     using object::object;
// };

#endif

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

#if defined(__cpp_nontype_template_parameter_class)
template <size_t N>
struct StringLiteral {
    constexpr StringLiteral(const char (&str)[N]) { std::copy_n(str, N, value); }
    char value[N];
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
    static constexpr auto name = const_name("tuple[")
                                 + ::pybind11::detail::concat(make_caster<Types>::name...)
                                 + const_name("]");
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
    static constexpr auto name
        = const_name("Callable[[") + ::pybind11::detail::concat(make_caster<Args>::name...)
          + const_name("], ") + make_caster<retval_type>::name + const_name("]");
};

#if defined(__cpp_nontype_template_parameter_class)
template <typing::StringLiteral lit>
struct handle_type_name<typing::TypeVar<lit>> {
    static constexpr auto name = const_name(lit.value);
};

template <typing::StringLiteral... lit>
struct handle_type_name<typing::LiteralStr<lit>> {
    static constexpr auto name
        = const_name("Literal[") + pybind11::detail::concat(lit.value) + const_name("]");
};

// template <int... intLit>
// struct handle_type_name<typing::Literal<intLit>> {
//     static constexpr auto name
//         = const_name("Literal[") + pybind11::detail::concat(intLit) + const_name("]");
// };

// template <bool... boolLit>
// struct handle_type_name<typing::Literal<boolLit>> {
//     static constexpr auto name
//         = const_name("Literal[") + pybind11::detail::concat(boolLit) + const_name("]");
// }

// template <>
// struct handle_type_name<typing::Literal<py::none>> {
//     static constexpr auto name
//         = const_name("Literal[None]");
// }

// template <any... anyLit>
// struct handle_type_name<typing::Literal<anyLit>> {
//     // TODO handle conststr
//     static constexpr auto name = const_name("Literal[") + pybind11::detail::concat(boolLit) + const_name("]");
// }
#endif

PYBIND11_NAMESPACE_END(detail) PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
template <typename T>
struct handle_type_name<typing::Type<T>> {
    static constexpr auto name = const_name("type[") + make_caster<T>::name + const_name("]");
};

template <typename... Types>
struct handle_type_name<typing::Union<Types...>> {
    static constexpr auto name = const_name("Union[")
                                 + ::pybind11::detail::concat(make_caster<Types>::name...)
                                 + const_name("]");
};

template <typename T>
struct handle_type_name<typing::Optional<T>> {
    static constexpr auto name = const_name("Optional[") + make_caster<T>::name + const_name("]");
};

#if defined(__cpp_nontype_template_parameter_class)
template <typing::StringLiteral StrLit>
struct handle_type_name<typing::TypeVar<StrLit>> {
    static constexpr auto name = const_name(StrLit.value);
};
#endif

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
