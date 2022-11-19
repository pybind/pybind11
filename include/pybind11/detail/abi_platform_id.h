// Copyright (c) 2022 The pybind Community.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include "common.h"

/// On MSVC, debug and release builds are not ABI-compatible!
#if defined(_MSC_VER) && defined(_DEBUG)
#    define PYBIND11_BUILD_TYPE "_debug"
#else
#    define PYBIND11_BUILD_TYPE ""
#endif

/// Let's assume that different compilers are ABI-incompatible.
/// A user can manually set this string if they know their
/// compiler is compatible.
#ifndef PYBIND11_COMPILER_TYPE
#    if defined(_MSC_VER)
#        define PYBIND11_COMPILER_TYPE "_msvc"
#    elif defined(__INTEL_COMPILER)
#        define PYBIND11_COMPILER_TYPE "_icc"
#    elif defined(__clang__)
#        define PYBIND11_COMPILER_TYPE "_clang"
#    elif defined(__PGI)
#        define PYBIND11_COMPILER_TYPE "_pgi"
#    elif defined(__MINGW32__)
#        define PYBIND11_COMPILER_TYPE "_mingw"
#    elif defined(__CYGWIN__)
#        define PYBIND11_COMPILER_TYPE "_gcc_cygwin"
#    elif defined(__GNUC__)
#        define PYBIND11_COMPILER_TYPE "_gcc"
#    else
#        define PYBIND11_COMPILER_TYPE "_unknown"
#    endif
#endif

/// Also standard libs
#ifndef PYBIND11_STDLIB
#    if defined(_LIBCPP_VERSION)
#        define PYBIND11_STDLIB "_libcpp"
#    elif defined(__GLIBCXX__) || defined(__GLIBCPP__)
#        define PYBIND11_STDLIB "_libstdcpp"
#    else
#        define PYBIND11_STDLIB ""
#    endif
#endif

/// On Linux/OSX, changes in __GXX_ABI_VERSION__ indicate ABI incompatibility.
#ifndef PYBIND11_BUILD_ABI
#    if defined(__GXX_ABI_VERSION)
#        define PYBIND11_BUILD_ABI "_cxxabi" PYBIND11_TOSTRING(__GXX_ABI_VERSION)
#    else
#        define PYBIND11_BUILD_ABI ""
#    endif
#endif

#ifndef PYBIND11_INTERNALS_KIND
#    if defined(WITH_THREAD)
#        define PYBIND11_INTERNALS_KIND ""
#    else
#        define PYBIND11_INTERNALS_KIND "_without_thread"
#    endif
#endif

/// See README_smart_holder.rst:
/// Classic / Conservative / Progressive cross-module compatibility
#ifndef PYBIND11_INTERNALS_SH_DEF
#    if defined(PYBIND11_USE_SMART_HOLDER_AS_DEFAULT)
#        define PYBIND11_INTERNALS_SH_DEF ""
#    else
#        define PYBIND11_INTERNALS_SH_DEF "_sh_def"
#    endif
#endif

/* NOTE - ATTENTION - WARNING - EXTREME CAUTION
   Changing this will break compatibility with `PYBIND11_INTERNALS_VERSION 4`
   See pybind11/detail/type_map.h for more information.
 */
#define PYBIND11_PLATFORM_ABI_ID_V4                                                               \
    PYBIND11_INTERNALS_KIND PYBIND11_COMPILER_TYPE PYBIND11_STDLIB PYBIND11_BUILD_ABI             \
        PYBIND11_BUILD_TYPE PYBIND11_INTERNALS_SH_DEF

/// LEGACY "ABI-breaking" APPROACH, ORIGINAL COMMENT
/// ------------------------------------------------
/// Tracks the `internals` and `type_info` ABI version independent of the main library version.
///
/// Some portions of the code use an ABI that is conditional depending on this
/// version number.  That allows ABI-breaking changes to be "pre-implemented".
/// Once the default version number is incremented, the conditional logic that
/// no longer applies can be removed.  Additionally, users that need not
/// maintain ABI compatibility can increase the version number in order to take
/// advantage of any functionality/efficiency improvements that depend on the
/// newer ABI.
///
/// WARNING: If you choose to manually increase the ABI version, note that
/// pybind11 may not be tested as thoroughly with a non-default ABI version, and
/// further ABI-incompatible changes may be made before the ABI is officially
/// changed to the new version.
#ifndef PYBIND11_INTERNALS_VERSION
#    define PYBIND11_INTERNALS_VERSION 4
#endif
