// Copyright (c) 2022 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include "descr.h"

#if defined(PYBIND11_ENABLE_TYPE_CASTER_ODR_GUARD)

#    if !defined(PYBIND11_CPP20) && defined(__GNUC__) && !defined(__clang__)
#        pragma GCC diagnostic ignored "-Wsubobject-linkage"
#    endif

#    include "../pytypes.h"
#    include "common.h"
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

using type_caster_odr_guard_registry_type = std::unordered_map<std::type_index, std::string>;

inline type_caster_odr_guard_registry_type &type_caster_odr_guard_registry() {
    // Using the no-destructor idiom (maximizes safety).
    static auto *reg = new type_caster_odr_guard_registry_type();
    return *reg;
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

// This macro is for cooperation with test_type_caster_odr_guard_?.cpp
#    ifndef PYBIND11_DETAIL_TYPE_CASTER_ODR_GUARD_THROW_DISABLED
#        define PYBIND11_DETAIL_TYPE_CASTER_ODR_GUARD_THROW_DISABLED false
#    endif

inline void type_caster_odr_guard_impl(const std::type_info &intrinsic_type_info,
                                       const src_loc &sloc,
                                       bool throw_disabled) {
    std::string source_file_line_from_sloc
        = std::string(sloc.file) + ':' + std::to_string(sloc.line);
// This macro is purely for debugging.
#    if defined(PYBIND11_DETAIL_TYPE_CASTER_ODR_GUARD_IMPL_DEBUG)
    // std::cout cannot be used here: static initialization could be incomplete.
    std::fprintf(stdout,
                 "\nTYPE_CASTER_ODR_GUARD_IMPL %s %s\n",
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
#    if defined(PYBIND11_DETAIL_TYPE_CASTER_ODR_GUARD_IMPL_DEBUG)
            std::fprintf(stderr, "\nDISABLED std::system_error: %s\n", msg.c_str());
            std::fflush(stderr);
#    endif
            type_caster_odr_violation_detected_counter()++;
        } else {
            throw std::system_error(std::make_error_code(std::errc::state_not_recoverable), msg);
        }
    }
}

namespace { // WARNING: This creates an ODR violation in the ODR guard itself,
            //          but we do not have any alternative at the moment.
// The ODR violation here does not involve any data at all.
// See also: Comment near top of descr.h & WARNING in descr.h

struct tu_local_no_data_always_false {
    explicit operator bool() const noexcept { return false; }
};

} // namespace

template <typename IntrinsicType, typename TypeCasterType>
struct type_caster_odr_guard : TypeCasterType {
    static tu_local_no_data_always_false translation_unit_local;

    type_caster_odr_guard() {
        // Possibly, good optimizers will elide this `if` (and below) completely.
        // It is needed only to trigger the TU-local mechanisms.
        if (translation_unit_local) {
        }
    }

    // The original author of this function is @amauryfa
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
          // Executed only once per process (e.g. when a PYBIND11_MODULE is initialized).
          // Conclusively tested vi test_type_caster_odr_guard_1, test_type_caster_odr_guard_2:
          // those tests will fail if the sloc here is not working as intended (TU-local).
          type_caster_odr_guard_impl(typeid(IntrinsicType),
                                     TypeCasterType::name.sloc,
                                     PYBIND11_DETAIL_TYPE_CASTER_ODR_GUARD_THROW_DISABLED);
          return tu_local_no_data_always_false();
      }();

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

#endif // PYBIND11_ENABLE_TYPE_CASTER_ODR_GUARD
