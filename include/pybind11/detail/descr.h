/*
    pybind11/detail/descr.h: Helper type for concatenating type signatures at compile time

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"

NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
NAMESPACE_BEGIN(detail)

/* Concatenate type signatures at compile time */
template <size_t Size1, size_t Size2> class descr {
    template <size_t Size1_, size_t Size2_> friend class descr;
public:
    constexpr descr() = default;

    constexpr descr(char const (&text) [Size1+1], const std::type_info * const (&types)[Size2+1])
        : descr(text, types,
                make_index_sequence<Size1>(),
                make_index_sequence<Size2>()) { }

    constexpr const char *text() const { return m_text; }
    constexpr const std::type_info * const * types() const { return m_types; }

    template <size_t OtherSize1, size_t OtherSize2>
    constexpr descr<Size1 + OtherSize1, Size2 + OtherSize2> operator+(const descr<OtherSize1, OtherSize2> &other) const {
        return concat(other,
                      make_index_sequence<Size1>(),
                      make_index_sequence<Size2>(),
                      make_index_sequence<OtherSize1>(),
                      make_index_sequence<OtherSize2>());
    }

protected:
    template <size_t... Indices1, size_t... Indices2>
    constexpr descr(
        char const (&text) [Size1+1],
        const std::type_info * const (&types) [Size2+1],
        index_sequence<Indices1...>, index_sequence<Indices2...>)
        : m_text{text[Indices1]..., '\0'},
          m_types{types[Indices2]...,  nullptr } {}

    template <size_t OtherSize1, size_t OtherSize2, size_t... Indices1,
              size_t... Indices2, size_t... OtherIndices1, size_t... OtherIndices2>
    constexpr descr<Size1 + OtherSize1, Size2 + OtherSize2>
    concat(const descr<OtherSize1, OtherSize2> &other,
           index_sequence<Indices1...>, index_sequence<Indices2...>,
           index_sequence<OtherIndices1...>, index_sequence<OtherIndices2...>) const {
        return descr<Size1 + OtherSize1, Size2 + OtherSize2>(
            { m_text[Indices1]..., other.m_text[OtherIndices1]..., '\0' },
            { m_types[Indices2]..., other.m_types[OtherIndices2]..., nullptr }
        );
    }

protected:
    char m_text[Size1 + 1];
    const std::type_info * m_types[Size2 + 1];
};

template <size_t Size> constexpr descr<Size - 1, 0> _(char const(&text)[Size]) {
    return descr<Size - 1, 0>(text, { nullptr });
}

template <size_t Rem, size_t... Digits> struct int_to_str : int_to_str<Rem/10, Rem%10, Digits...> { };
template <size_t...Digits> struct int_to_str<0, Digits...> {
    static constexpr auto digits = descr<sizeof...(Digits), 0>({ ('0' + Digits)..., '\0' }, { nullptr });
};

// Ternary description (like std::conditional)
template <bool B, size_t Size1, size_t Size2>
constexpr enable_if_t<B, descr<Size1 - 1, 0>> _(char const(&text1)[Size1], char const(&)[Size2]) {
    return _(text1);
}
template <bool B, size_t Size1, size_t Size2>
constexpr enable_if_t<!B, descr<Size2 - 1, 0>> _(char const(&)[Size1], char const(&text2)[Size2]) {
    return _(text2);
}
template <bool B, size_t SizeA1, size_t SizeA2, size_t SizeB1, size_t SizeB2>
constexpr enable_if_t<B, descr<SizeA1, SizeA2>> _(descr<SizeA1, SizeA2> d, descr<SizeB1, SizeB2>) { return d; }
template <bool B, size_t SizeA1, size_t SizeA2, size_t SizeB1, size_t SizeB2>
constexpr enable_if_t<!B, descr<SizeB1, SizeB2>> _(descr<SizeA1, SizeA2>, descr<SizeB1, SizeB2> d) { return d; }

template <size_t Size> auto constexpr _() -> decltype(int_to_str<Size / 10, Size % 10>::digits) {
    return int_to_str<Size / 10, Size % 10>::digits;
}

template <typename Type> constexpr descr<1, 1> _() {
    return descr<1, 1>({ '%', '\0' }, { &typeid(Type), nullptr });
}

constexpr descr<0, 0> concat() { return _(""); }

template <size_t Size1, size_t Size2>
constexpr descr<Size1, Size2> concat(descr<Size1, Size2> descr) { return descr; }

template <size_t Size1, size_t Size2, typename... Args>
constexpr auto concat(descr<Size1, Size2> d, Args... args)
    -> decltype(descr<Size1 + 2, Size2>{} + concat(args...)) {
    return d + _(", ") + concat(args...);
}

template <size_t Size1, size_t Size2>
constexpr descr<Size1 + 2, Size2> type_descr(descr<Size1, Size2> descr) {
    return _("{") + descr + _("}");
}

NAMESPACE_END(detail)
NAMESPACE_END(PYBIND11_NAMESPACE)
