// Copyright (c) 2022 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#if !defined(PYBIND11_TYPE_CASTER_ODR_GUARD_ON) && !defined(PYBIND11_TYPE_CASTER_ODR_GUARD_OFF)   \
    && (defined(_MSC_VER) || defined(PYBIND11_CPP20)                                              \
        || (defined(PYBIND11_CPP17) /* && defined(__clang__)*/))
#    define PYBIND11_TYPE_CASTER_ODR_GUARD_ON
#endif

#ifndef PYBIND11_TYPE_CASTER_ODR_GUARD_ON

#    define PYBIND11_TYPE_CASTER_SOURCE_FILE_LINE

#    define PYBIND11_DETAIL_TYPE_CASTER_ACCESS_TRANSLATION_UNIT_LOCAL(...)

#else

#    if defined(__GNUC__) && !defined(PYBIND11_CPP20)
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
#    include <unordered_map>
#    include <utility>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

inline std::unordered_map<std::type_index, std::string> &odr_guard_registry() {
    static std::unordered_map<std::type_index, std::string> reg;
    return reg;
}

inline const char *source_file_line_basename(const char *sfl) {
    unsigned i_base = 0;
    for (unsigned i = 0; sfl[i] != '\0'; i++) {
        if (sfl[i] == '/' || sfl[i] == '\\') {
            i_base = i + 1;
        }
    }
    return sfl + i_base;
}

#    ifndef PYBIND11_DETAIL_ODR_GUARD_IMPL_THROW_DISABLED
#        define PYBIND11_DETAIL_ODR_GUARD_IMPL_THROW_DISABLED false
#    endif

template <typename IntrinsicType>
void odr_guard_impl(const std::type_index &it_ti,
                    const char *source_file_line,
                    bool throw_disabled) {
    // std::cout cannot be used here: static initialization could be incomplete.
#    define PYBIND11_DETAIL_ODR_GUARD_IMPL_PRINTF_OFF
#    ifdef PYBIND11_DETAIL_ODR_GUARD_IMPL_PRINTF_ON
    std::fprintf(
        stdout, "\nODR_GUARD_IMPL %s %s\n", type_id<IntrinsicType>().c_str(), source_file_line);
    std::fflush(stdout);
#    endif
    std::string sflbn_str{source_file_line_basename(source_file_line)};
    auto ins = odr_guard_registry().insert({it_ti, source_file_line});
    auto reg_iter = ins.first;
    auto added = ins.second;
    if (!added
        && strcmp(source_file_line_basename(reg_iter->second.c_str()),
                  source_file_line_basename(source_file_line))
               != 0) {
        std::string msg("ODR VIOLATION DETECTED: pybind11::detail::type_caster<"
                        + type_id<IntrinsicType>() + ">: SourceLocation1=\"" + reg_iter->second
                        + "\", SourceLocation2=\"" + source_file_line + "\"");
        if (throw_disabled) {
            std::fprintf(stderr, "\nDISABLED std::system_error: %s\n", msg.c_str());
            std::fflush(stderr);
        } else {
            throw std::system_error(std::make_error_code(std::errc::state_not_recoverable), msg);
        }
    }
}

namespace {

template <size_t N, typename... Ts>
struct tu_local_descr : descr<N, Ts...> {
    using descr_t = descr<N, Ts...>;
    using descr_t::descr_t;
};

template <size_t N>
constexpr tu_local_descr<N - 1> tu_local_const_name(char const (&text)[N]) {
    return tu_local_descr<N - 1>(text);
}
constexpr tu_local_descr<0> tu_local_const_name(char const (&)[1]) { return {}; }

} // namespace

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

#    define PYBIND11_TYPE_CASTER_SOURCE_FILE_LINE                                                 \
        static constexpr auto source_file_line                                                    \
            = ::pybind11::detail::tu_local_const_name(__FILE__ ":" PYBIND11_TOSTRING(__LINE__));

#    define PYBIND11_DETAIL_TYPE_CASTER_ACCESS_TRANSLATION_UNIT_LOCAL(...)                        \
        if (::pybind11::detail::make_caster<__VA_ARGS__>::translation_unit_local) {               \
        }

#endif
