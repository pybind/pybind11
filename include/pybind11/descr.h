/*
    pybind11/descr.h: Helper type for concatenating type signatures
    either at runtime (C++11) or compile time (C++14)

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

#if defined(__INTEL_COMPILER)
/* C++14 features not supported for now */
#elif defined(__clang__)
#  if __has_feature(cxx_return_type_deduction) && __has_feature(cxx_relaxed_constexpr)
#    define PYBIND11_CPP14
#  endif
#elif defined(__GNUG__)
#  if __cpp_constexpr >= 201304 && __cpp_decltype_auto >= 201304
#    define PYBIND11_CPP14
#  endif
#endif

// Character used to separate in/out arguments; will be replaced with " -> ".  (We don't just look
// for " -> " because that can also be in a function argument name).
#define PYBIND11_DESCR_IN_OUT_SEP '`'
// Character that goes in front of arguments (used in the tuple caster)
#define PYBIND11_DESCR_ARG_PREFIX '$'
// Character used to signal that the current type has an associated type_info pointer
#define PYBIND11_DESCR_TYPE_INFO '~'
// Character that signals that the type name should be looked up from its type_info
#define PYBIND11_DESCR_AUTO_TYPE_NAME '%'
// The above will be ordered in the order listed above, i.e. ${~%} indicates varaible with a type_info
// record that should have its human-readable name derived from that type info.

#if defined(PYBIND11_CPP14) /* Concatenate type signatures at compile time using C++14 */

template <size_t Size1, size_t Size2> class descr {
    template <size_t Size1_, size_t Size2_> friend class descr;
public:
    constexpr descr(char const (&text) [Size1+1], const std::type_info * const (&types)[Size2+1])
        : descr(text, types,
                typename make_index_sequence<Size1>::type(),
                typename make_index_sequence<Size2>::type()) { }

    constexpr const char *text() const { return m_text; }
    constexpr const std::type_info * const * types() const { return m_types; }

    template <size_t OtherSize1, size_t OtherSize2>
    constexpr descr<Size1 + OtherSize1, Size2 + OtherSize2> operator+(const descr<OtherSize1, OtherSize2> &other) const {
        return concat(other,
                      typename make_index_sequence<Size1>::type(),
                      typename make_index_sequence<Size2>::type(),
                      typename make_index_sequence<OtherSize1>::type(),
                      typename make_index_sequence<OtherSize2>::type());
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
// Same as above, but holds the type and prefixes with & (so '&%' is a generic type with name
// determined from its typeid, '&blah' is some type with display name 'blah').
template <typename Type, size_t Size> constexpr descr<Size, 1> _(char const(&text)[Size]) {
    return descr<1, 1>({ PYBIND11_DESCR_TYPE_INFO, '\0' }, { &typeid(Type), nullptr }) + _(text);
}
// Single-char versions of the above two:
inline constexpr descr<1, 0> _(const char c) { return descr<1, 0>({ c, '\0' }, { nullptr }); }
template <typename Type> constexpr descr<1, 1> _(const char c) { return _<Type>({ c, '\0' }); }

template <size_t Rem, size_t... Digits> struct int_to_str : int_to_str<Rem/10, Rem%10, Digits...> { };
template <size_t...Digits> struct int_to_str<0, Digits...> {
    static constexpr auto digits = descr<sizeof...(Digits), 0>({ ('0' + Digits)..., '\0' }, { nullptr });
};

// Ternary description (like std::conditional)
template <bool B, size_t Size1, size_t Size2>
constexpr typename std::enable_if<B, descr<Size1 - 1, 0>>::type _(char const(&text1)[Size1], char const(&)[Size2]) {
    return _(text1);
}
template <bool B, size_t Size1, size_t Size2>
constexpr typename std::enable_if<!B, descr<Size2 - 1, 0>>::type _(char const(&)[Size1], char const(&text2)[Size2]) {
    return _(text2);
}

template <size_t Size> auto constexpr _() -> decltype(int_to_str<Size / 10, Size % 10>::digits) {
    return int_to_str<Size / 10, Size % 10>::digits;
}

template <typename Type> constexpr descr<2, 1> _() {
    return descr<2, 1>({ PYBIND11_DESCR_TYPE_INFO, PYBIND11_DESCR_AUTO_TYPE_NAME, '\0' }, { &typeid(Type), nullptr });
}

inline constexpr descr<0, 0> concat() { return _(""); }
template <size_t Size1, size_t Size2, typename... Args> auto constexpr concat(descr<Size1, Size2> descr) { return descr; }
template <size_t Size1, size_t Size2, typename... Args> auto constexpr concat(descr<Size1, Size2> descr, Args&&... args) { return descr + _(", ") + concat(args...); }
template <size_t Size1, size_t Size2> auto constexpr type_descr(descr<Size1, Size2> descr) { return _("{") + descr + _("}"); }

#define PYBIND11_DESCR constexpr auto

#else /* Simpler C++11 implementation based on run-time memory allocation and copying */

class descr {
public:
    PYBIND11_NOINLINE descr(const char *text, const std::type_info * const * types) {
        size_t nChars = len(text), nTypes = len(types);
        m_text  = new char[nChars];
        m_types = new const std::type_info *[nTypes];
        memcpy(m_text, text, nChars * sizeof(char));
        memcpy(m_types, types, nTypes * sizeof(const std::type_info *));
    }

    PYBIND11_NOINLINE descr friend operator+(descr &&d1, descr &&d2) {
        descr r;

        size_t nChars1 = len(d1.m_text), nTypes1 = len(d1.m_types);
        size_t nChars2 = len(d2.m_text), nTypes2 = len(d2.m_types);

        r.m_text  = new char[nChars1 + nChars2 - 1];
        r.m_types = new const std::type_info *[nTypes1 + nTypes2 - 1];
        memcpy(r.m_text, d1.m_text, (nChars1-1) * sizeof(char));
        memcpy(r.m_text + nChars1 - 1, d2.m_text, nChars2 * sizeof(char));
        memcpy(r.m_types, d1.m_types, (nTypes1-1) * sizeof(std::type_info *));
        memcpy(r.m_types + nTypes1 - 1, d2.m_types, nTypes2 * sizeof(std::type_info *));

        delete[] d1.m_text; delete[] d1.m_types;
        delete[] d2.m_text; delete[] d2.m_types;

        return r;
    }

    char *text() { return m_text; }
    const std::type_info * * types() { return m_types; }

protected:
    PYBIND11_NOINLINE descr() { }

    template <typename T> static size_t len(const T *ptr) { // return length including null termination
        const T *it = ptr;
        while (*it++ != (T) 0)
            ;
        return static_cast<size_t>(it - ptr);
    }

    const std::type_info **m_types = nullptr;
    char *m_text = nullptr;
};

/* The 'PYBIND11_NOINLINE inline' combinations below are intentional to get the desired linkage while producing as little object code as possible */

PYBIND11_NOINLINE inline descr _(const char *text) {
    const std::type_info *types[1] = { nullptr };
    return descr(text, types);
}

PYBIND11_NOINLINE inline descr _(char c) {
    // Can't pass this directly into the argument before C++14 (C++11 treats it as an initializer_list)
    const char c0[2] = {c, '\0'};
    const std::type_info *types[1] = { nullptr };
    return descr(c0, types);
}

template <typename Type> PYBIND11_NOINLINE descr _(char c) {
    const char c0[3] = {PYBIND11_DESCR_TYPE_INFO, c, '\0'};
    const std::type_info *types[2] = { &typeid(Type), nullptr };
    return descr(c0, types);
}

template <typename Type> PYBIND11_NOINLINE descr _(const char *text) {
    const char c0[2] = {PYBIND11_DESCR_TYPE_INFO, '\0'};
    const std::type_info *types[2] = { &typeid(Type), nullptr };
    const std::type_info *notypes[1] = { nullptr };
    return descr(c0, types) + descr(text, notypes);
}

template <bool B> PYBIND11_NOINLINE typename std::enable_if<B, descr>::type _(const char *text1, const char *) { return _(text1); }
template <bool B> PYBIND11_NOINLINE typename std::enable_if<!B, descr>::type _(char const *, const char *text2) { return _(text2); }

template <typename Type> descr _() { return _<Type>(PYBIND11_DESCR_AUTO_TYPE_NAME); }

template <size_t Size> PYBIND11_NOINLINE descr _() {
    const std::type_info *types[1] = { nullptr };
    return descr(std::to_string(Size).c_str(), types);
}

PYBIND11_NOINLINE inline descr concat() { return _(""); }
PYBIND11_NOINLINE inline descr concat(descr &&d) { return d; }
template <typename... Args> PYBIND11_NOINLINE descr concat(descr &&d, Args&&... args) { return std::move(d) + _(", ") + concat(std::forward<Args>(args)...); }
PYBIND11_NOINLINE inline descr type_descr(descr&& d) { return _("{") + std::move(d) + _("}"); }

#define PYBIND11_DESCR ::pybind11::detail::descr
#endif

NAMESPACE_END(detail)
NAMESPACE_END(pybind11)
