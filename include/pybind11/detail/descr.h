// Copyright (c) 2022 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
/*
    pybind11/detail/descr.h: Helper type for concatenating type signatures at compile time

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

#if !defined(_MSC_VER)
#    define PYBIND11_DESCR_CONSTEXPR static constexpr
#else
#    define PYBIND11_DESCR_CONSTEXPR const
#endif

// type_caster_odr_guard.h requires Translation-Unit-local features
// (https://en.cppreference.com/w/cpp/language/tu_local), standardized only with C++20.
// The ODR guard creates ODR violations itself (see WARNINGs below & in
// type_caster_odr_guard.h), but the dedicated test_type_caster_odr_guard_1,
// test_type_caster_odr_guard_2 pair of unit tests passes reliably with almost all
// tested C++17 & C++20 compilers, and even the exceptions are not due to ODR issues:
// * MSVC 2017 does not support __builtin_FILE(), __builtin_LINE().
// * Intel 2021.6.0.20220226 (g++ 9.4 mode) __builtin_LINE() is unreliable
//   (line numbers vary between translation units).
// Here we want to test the ODR guard in as many environments as possible, but
// it is NOT recommended to turn on the guard in regular builds, production, or
// debug. The guard is meant to be used similar to a sanitizer, to check for type_caster
// ODR violations in binaries that are otherwise already fully tested and assumed to be healthy.
#if defined(PYBIND11_TYPE_CASTER_ODR_GUARD_ON_IF_AVAILABLE)                                       \
    && !defined(PYBIND11_TYPE_CASTER_ODR_GUARD_ON) && !defined(__INTEL_COMPILER)                  \
    && ((defined(_MSC_VER) && _MSC_VER >= 1920) || defined(PYBIND11_CPP17))
#    define PYBIND11_TYPE_CASTER_ODR_GUARD_ON
#endif

#ifdef PYBIND11_TYPE_CASTER_ODR_GUARD_ON

// struct src_loc supports type_caster_odr_guard.h

// Not using std::source_location because:
// 1. "It is unspecified whether the copy/move constructors and the copy/move
//    assignment operators of source_location are trivial and/or constexpr."
//    (https://en.cppreference.com/w/cpp/utility/source_location).
// 2. A matching no-op stub is needed (below) to avoid code duplication.
struct src_loc {
    const char *file;
    unsigned line;

    constexpr src_loc(const char *file, unsigned line) : file(file), line(line) {}

    static constexpr src_loc here(const char *file = __builtin_FILE(),
                                  unsigned line = __builtin_LINE()) {
        return src_loc(file, line);
    }

    constexpr src_loc if_known_or(const src_loc &other) const {
        if (file != nullptr) {
            return *this;
        }
        return other;
    }
};

#else

// No-op stub, to avoid code duplication, expected to be optimized out completely.
struct src_loc {
    constexpr src_loc(const char *, unsigned) {}

    static constexpr src_loc here(const char * = nullptr, unsigned = 0) {
        return src_loc(nullptr, 0);
    }

    constexpr src_loc if_known_or(const src_loc &) const { return *this; }
};

#endif

#ifdef PYBIND11_TYPE_CASTER_ODR_GUARD_ON
namespace { // WARNING: This creates an ODR violation in the ODR guard itself,
            //          but we do not have anything better at the moment.
// The ODR violation here is a difference in constexpr between multiple TUs.
// All definitions have the same data layout, the only difference is the
// text const char* pointee (the pointees are identical in value),
// src_loc const char* file pointee (the pointees are different in value),
// src_loc unsigned line value.
// See also: Comment above; WARNING in type_caster_odr_guard.h
#endif

/* Concatenate type signatures at compile time */
template <size_t N, typename... Ts>
struct descr {
    char text[N + 1]{'\0'};
    src_loc sloc;

    explicit constexpr descr(src_loc sloc) : sloc(sloc) {}
    // NOLINTNEXTLINE(google-explicit-constructor)
    constexpr descr(char const (&s)[N + 1], src_loc sloc = src_loc::here())
        : descr(s, make_index_sequence<N>(), sloc) {}

    template <size_t... Is>
    constexpr descr(char const (&s)[N + 1], index_sequence<Is...>, src_loc sloc = src_loc::here())
        : text{s[Is]..., '\0'}, sloc(sloc) {}

    template <typename... Chars>
    // NOLINTNEXTLINE(google-explicit-constructor)
    constexpr descr(src_loc sloc, char c, Chars... cs)
        : text{c, static_cast<char>(cs)..., '\0'}, sloc(sloc) {}

    static constexpr std::array<const std::type_info *, sizeof...(Ts) + 1> types() {
        return {{&typeid(Ts)..., nullptr}};
    }
};

template <size_t N1, size_t N2, typename... Ts1, typename... Ts2, size_t... Is1, size_t... Is2>
constexpr descr<N1 + N2, Ts1..., Ts2...> plus_impl(const descr<N1, Ts1...> &a,
                                                   const descr<N2, Ts2...> &b,
                                                   index_sequence<Is1...>,
                                                   index_sequence<Is2...>) {
    PYBIND11_WORKAROUND_INCORRECT_MSVC_C4100(b);
    return descr<N1 + N2, Ts1..., Ts2...>{
        a.sloc.if_known_or(b.sloc), a.text[Is1]..., b.text[Is2]...};
}

template <size_t N1, size_t N2, typename... Ts1, typename... Ts2>
constexpr descr<N1 + N2, Ts1..., Ts2...> operator+(const descr<N1, Ts1...> &a,
                                                   const descr<N2, Ts2...> &b) {
    return plus_impl(a, b, make_index_sequence<N1>(), make_index_sequence<N2>());
}

template <size_t N>
constexpr descr<N - 1> const_name(char const (&text)[N], src_loc sloc = src_loc::here()) {
    return descr<N - 1>(text, sloc);
}
constexpr descr<0> const_name(char const (&)[1], src_loc sloc = src_loc::here()) {
    return descr<0>(sloc);
}

template <size_t Rem, size_t... Digits>
struct int_to_str : int_to_str<Rem / 10, Rem % 10, Digits...> {};
template <size_t... Digits>
struct int_to_str<0, Digits...> {
    // WARNING: This only works with C++17 or higher.
    // src_loc not tracked (not needed in this situation, at least at the moment).
    static constexpr auto digits
        = descr<sizeof...(Digits)>(src_loc{nullptr, 0}, ('0' + Digits)...);
};

// Ternary description (like std::conditional)
template <bool B, size_t N1, size_t N2>
constexpr enable_if_t<B, descr<N1 - 1>>
const_name(char const (&text1)[N1], char const (&)[N2], src_loc sloc = src_loc::here()) {
    return const_name(text1, sloc);
}
template <bool B, size_t N1, size_t N2>
constexpr enable_if_t<!B, descr<N2 - 1>>
const_name(char const (&)[N1], char const (&text2)[N2], src_loc sloc = src_loc::here()) {
    return const_name(text2, sloc);
}

template <bool B, typename T1, typename T2>
constexpr enable_if_t<B, T1> const_name(const T1 &d, const T2 &) {
    return d;
}
template <bool B, typename T1, typename T2>
constexpr enable_if_t<!B, T2> const_name(const T1 &, const T2 &d) {
    return d;
}

template <size_t Size>
auto constexpr const_name() -> remove_cv_t<decltype(int_to_str<Size / 10, Size % 10>::digits)> {
    // src_loc not tracked (not needed in this situation, at least at the moment).
    return int_to_str<Size / 10, Size % 10>::digits;
}

template <typename Type>
constexpr descr<1, Type> const_name(src_loc sloc = src_loc::here()) {
    return {sloc, '%'};
}

// If "_" is defined as a macro, py::detail::_ cannot be provided.
// It is therefore best to use py::detail::const_name universally.
// This block is for backward compatibility only.
// (The const_name code is repeated to avoid introducing a "_" #define ourselves.)
#ifndef _
#    define PYBIND11_DETAIL_UNDERSCORE_BACKWARD_COMPATIBILITY
template <size_t N>
constexpr descr<N - 1> _(char const (&text)[N], src_loc sloc = src_loc::here()) {
    return const_name<N>(text, sloc);
}
template <bool B, size_t N1, size_t N2>
constexpr enable_if_t<B, descr<N1 - 1>>
_(char const (&text1)[N1], char const (&text2)[N2], src_loc sloc = src_loc::here()) {
    return const_name<B, N1, N2>(text1, text2, sloc);
}
template <bool B, size_t N1, size_t N2>
constexpr enable_if_t<!B, descr<N2 - 1>>
_(char const (&text1)[N1], char const (&text2)[N2], src_loc sloc = src_loc::here()) {
    return const_name<B, N1, N2>(text1, text2, sloc);
}
template <bool B, typename T1, typename T2>
constexpr enable_if_t<B, T1> _(const T1 &d1, const T2 &d2) {
    return const_name<B, T1, T2>(d1, d2);
}
template <bool B, typename T1, typename T2>
constexpr enable_if_t<!B, T2> _(const T1 &d1, const T2 &d2) {
    return const_name<B, T1, T2>(d1, d2);
}

template <size_t Size>
auto constexpr _() -> remove_cv_t<decltype(int_to_str<Size / 10, Size % 10>::digits)> {
    // src_loc not tracked (not needed in this situation, at least at the moment).
    return const_name<Size>();
}
template <typename Type>
constexpr descr<1, Type> _(src_loc sloc = src_loc::here()) {
    return const_name<Type>(sloc);
}
#endif // #ifndef _

constexpr descr<0> concat(src_loc sloc = src_loc::here()) { return descr<0>{sloc}; }

template <size_t N, typename... Ts>
constexpr descr<N, Ts...> concat(const descr<N, Ts...> &descr) {
    return descr;
}

template <size_t N, typename... Ts, typename... Args>
constexpr auto concat(const descr<N, Ts...> &d, const Args &...args)
    -> decltype(std::declval<descr<N + 2, Ts...>>() + concat(args...)) {
    // Ensure that src_loc of existing descr is used.
    return d + const_name(", ", src_loc{nullptr, 0}) + concat(args...);
}

template <size_t N, typename... Ts>
constexpr descr<N + 2, Ts...> type_descr(const descr<N, Ts...> &descr) {
    // Ensure that src_loc of existing descr is used.
    return const_name("{", src_loc{nullptr, 0}) + descr + const_name("}");
}

#ifdef PYBIND11_TYPE_CASTER_ODR_GUARD_ON
} // namespace
#endif

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
