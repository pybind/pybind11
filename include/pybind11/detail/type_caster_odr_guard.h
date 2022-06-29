// Copyright (c) 2022 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

// The type_caster ODR guard feature requires Translation-Unit-local entities
// (https://en.cppreference.com/w/cpp/language/tu_local), a C++20 feature, but
// all tested C++17 compilers support this feature already.
#if !defined(PYBIND11_TYPE_CASTER_ODR_GUARD_ON) && !defined(PYBIND11_TYPE_CASTER_ODR_GUARD_OFF)   \
    && ((defined(_MSC_VER) && _MSC_VER >= 1920) || defined(PYBIND11_CPP17))
#    define PYBIND11_TYPE_CASTER_ODR_GUARD_ON
#endif

#ifndef PYBIND11_TYPE_CASTER_ODR_GUARD_ON

#    define PYBIND11_TYPE_CASTER_SOURCE_FILE_LINE

#    define PYBIND11_DETAIL_TYPE_CASTER_ACCESS_TRANSLATION_UNIT_LOCAL(...)

#else

#    if !defined(PYBIND11_CPP20) && defined(__GNUC__) && !defined(__clang__)                      \
        && !defined(__INTEL_COMPILER)
#        pragma GCC diagnostic ignored "-Wsubobject-linkage"
#    endif

#    include "common.h"
#    include "descr.h"
#    include "typeid.h"

#    include <cstdio>
#    include <cstring>
#    include <string>
#    include <system_error>
#    include <typeindex>
#    include <typeinfo>
#    include <unordered_map>
#    include <utility>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

struct src_loc {
    const char *file;
    unsigned line;

    // constexpr src_loc() : file(nullptr), line(0) {}
    constexpr src_loc(const char *file, unsigned line) : file(file), line(line) {}

    static constexpr src_loc here(const char *file = __builtin_FILE(),
                                  unsigned line = __builtin_LINE()) {
        return src_loc(file, line);
    }
};

inline std::unordered_map<std::type_index, std::string> &type_caster_odr_guard_registry() {
    static std::unordered_map<std::type_index, std::string> reg;
    return reg;
}

inline unsigned &type_caster_odr_violation_detected_counter() {
    static unsigned counter = 0;
    return counter;
}

inline std::string source_file_line_basename(const char *sfl) {
    unsigned i_base = 0;
    for (unsigned i = 0; sfl[i] != '\0'; i++) {
        if (sfl[i] == '/' || sfl[i] == '\\') {
            i_base = i + 1;
        }
    }
    return std::string(sfl + i_base);
}

#    ifndef PYBIND11_DETAIL_TYPE_CASTER_ODR_GUARD_IMPL_THROW_DISABLED
#        define PYBIND11_DETAIL_TYPE_CASTER_ODR_GUARD_IMPL_THROW_DISABLED false
#    endif

inline void type_caster_odr_guard_impl(const std::type_info &intrinsic_type_info,
                                       const char *source_file_line_from_macros,
                                       const src_loc &sloc,
                                       bool throw_disabled) {
    // std::cout cannot be used here: static initialization could be incomplete.
#    define PYBIND11_DETAIL_TYPE_CASTER_ODR_GUARD_IMPL_PRINTF_ON
#    ifdef PYBIND11_DETAIL_TYPE_CASTER_ODR_GUARD_IMPL_PRINTF_ON
    std::fprintf(stdout,
                 "\nTYPE_CASTER_ODR_GUARD_IMPL %s %s\n",
                 clean_type_id(intrinsic_type_info.name()).c_str(),
                 source_file_line_from_macros);
    std::string source_file_line_from_sloc
        = std::string(sloc.file) + ':' + std::to_string(sloc.line);
    std::fprintf(stdout,
                 "%s %s %s\n",
                 (source_file_line_from_sloc == source_file_line_from_macros
                      ? "                 SLOC_SAME"
                      : "                 SLOC_DIFF"),
                 clean_type_id(intrinsic_type_info.name()).c_str(),
                 source_file_line_from_sloc.c_str());
    std::fflush(stdout);
#    endif
    auto ins = type_caster_odr_guard_registry().insert(
        {std::type_index(intrinsic_type_info), source_file_line_from_sloc});
    auto reg_iter = ins.first;
    auto added = ins.second;
    if (!added
        && source_file_line_basename(reg_iter->second.c_str())
               != source_file_line_basename(source_file_line_from_sloc.c_str())) {
        std::string msg("ODR VIOLATION DETECTED: pybind11::detail::type_caster<"
                        + clean_type_id(intrinsic_type_info.name()) + ">: SourceLocation1=\""
                        + reg_iter->second + "\", SourceLocation2=\"" + source_file_line_from_sloc
                        + "\"");
        if (throw_disabled) {
            std::fprintf(stderr, "\nDISABLED std::system_error: %s\n", msg.c_str());
            std::fflush(stderr);
            type_caster_odr_violation_detected_counter()++;
        } else {
            throw std::system_error(std::make_error_code(std::errc::state_not_recoverable), msg);
        }
    }
}

namespace {

template <size_t N, typename... Ts>
struct tu_local_descr {
    char text[N + 1]{'\0'};
    src_loc sloc;

    explicit constexpr tu_local_descr(src_loc sloc = src_loc::here()) : sloc(sloc) {}
    // NOLINTNEXTLINE(google-explicit-constructor)
    constexpr tu_local_descr(char const (&s)[N + 1], src_loc sloc = src_loc::here())
        : tu_local_descr(s, make_index_sequence<N>(), sloc) {}

    template <size_t... Is>
    constexpr tu_local_descr(char const (&s)[N + 1],
                             index_sequence<Is...>,
                             src_loc sloc = src_loc::here())
        : text{s[Is]..., '\0'}, sloc(sloc) {}

    template <typename... Chars>
    // NOLINTNEXTLINE(google-explicit-constructor)
    constexpr tu_local_descr(char c, Chars... cs, src_loc sloc = src_loc::here())
        : text{c, static_cast<char>(cs)..., '\0'}, sloc(sloc) {}
};

template <size_t N>
constexpr tu_local_descr<N - 1> tu_local_const_name(char const (&text)[N],
                                                    src_loc sloc = src_loc::here()) {
    return tu_local_descr<N - 1>(text, sloc);
}
constexpr tu_local_descr<0> tu_local_const_name(char const (&)[1],
                                                src_loc sloc = src_loc::here()) {
    return tu_local_descr<0>(sloc);
}

struct tu_local_no_data_always_false {
    explicit operator bool() const noexcept { return false; }
};

} // namespace

#    ifndef PYBIND11_TYPE_CASTER_ODR_GUARD_STRICT
#        define PYBIND11_TYPE_CASTER_ODR_GUARD_STRICT
#    endif

template <typename TypeCasterType, typename SFINAE = void>
struct get_type_caster_source_file_line {
#    ifdef PYBIND11_TYPE_CASTER_ODR_GUARD_STRICT
    static_assert(TypeCasterType::source_file_line,
                  "PYBIND11_TYPE_CASTER_SOURCE_FILE_LINE is MISSING: Please add that macro to the "
                  "TypeCasterType, or undefine PYBIND11_TYPE_CASTER_ODR_GUARD_STRICT");
#    else
    static constexpr auto source_file_line = tu_local_const_name("UNAVAILABLE");
#    endif
};

template <typename TypeCasterType>
struct get_type_caster_source_file_line<
    TypeCasterType,
    enable_if_t<std::is_class<decltype(TypeCasterType::source_file_line)>::value>> {
    static constexpr auto source_file_line = TypeCasterType::source_file_line;
};

template <typename IntrinsicType, typename TypeCasterType>
struct type_caster_odr_guard : TypeCasterType {
    static tu_local_no_data_always_false translation_unit_local;

    type_caster_odr_guard() {
        if (translation_unit_local) {
        }
    }

    template <typename CType, typename... Arg>
    static handle cast(CType &&src, return_value_policy policy, handle parent, Arg &&...arg) {
        if (translation_unit_local) {
        }
        return TypeCasterType::cast(
            std::forward<CType>(src), policy, parent, std::forward<Arg>(arg)...);
    }
};

template <typename IntrinsicType, typename TypeCasterType>
tu_local_no_data_always_false
    type_caster_odr_guard<IntrinsicType, TypeCasterType>::translation_unit_local
    = []() {
          type_caster_odr_guard_impl(
              typeid(IntrinsicType),
              get_type_caster_source_file_line<TypeCasterType>::source_file_line.text,
              get_type_caster_source_file_line<TypeCasterType>::source_file_line.sloc,
              PYBIND11_DETAIL_TYPE_CASTER_ODR_GUARD_IMPL_THROW_DISABLED);
          return tu_local_no_data_always_false();
      }();

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

#    define PYBIND11_TYPE_CASTER_SOURCE_FILE_LINE                                                 \
        static constexpr auto source_file_line                                                    \
            = ::pybind11::detail::tu_local_const_name(__FILE__ ":" PYBIND11_TOSTRING(__LINE__));

#    define PYBIND11_DETAIL_TYPE_CASTER_ACCESS_TRANSLATION_UNIT_LOCAL(...)

#endif
