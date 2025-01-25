#pragma once

// Copyright (c) 2024 The pybind Community.

// To maximize reusability:
// DO NOT ADD CODE THAT REQUIRES C++ EXCEPTION HANDLING.

#include "wrap_include_python_h.h"

// Implementation details. DO NOT USE ELSEWHERE. (Unfortunately we cannot #undef them.)
// This is duplicated here to maximize portability.
#define PYBIND11_PLATFORM_ABI_ID_STRINGIFY(x) #x
#define PYBIND11_PLATFORM_ABI_ID_TOSTRING(x) PYBIND11_PLATFORM_ABI_ID_STRINGIFY(x)

// On MSVC, debug and release builds are not ABI-compatible!
#if defined(_MSC_VER) && defined(_DEBUG)
#    define PYBIND11_BUILD_TYPE "_debug"
#else
#    define PYBIND11_BUILD_TYPE ""
#endif

// Let's assume that different compilers are ABI-incompatible.
// A user can manually set this string if they know their
// compiler is compatible.
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

// Also standard libs
#ifndef PYBIND11_STDLIB
#    if defined(_LIBCPP_VERSION)
#        define PYBIND11_STDLIB "_libcpp"
#    elif defined(__GLIBCXX__) || defined(__GLIBCPP__)
#        define PYBIND11_STDLIB "_libstdcpp"
#    else
#        define PYBIND11_STDLIB ""
#    endif
#endif

#ifndef PYBIND11_BUILD_ABI
#    if defined(__GXX_ABI_VERSION) // Linux/OSX.
#        define PYBIND11_BUILD_ABI "_cxxabi" PYBIND11_PLATFORM_ABI_ID_TOSTRING(__GXX_ABI_VERSION)
#    elif defined(_MSC_VER)               // See PR #4953.
#        if defined(_MT) && defined(_DLL) // Corresponding to CL command line options /MD or /MDd.
#            if (_MSC_VER) / 100 == 19
#                define PYBIND11_BUILD_ABI "_md_mscver19"
#            else
#                error "Unknown major version for MSC_VER: PLEASE REVISE THIS CODE."
#            endif
#        elif defined(_MT) // Corresponding to CL command line options /MT or /MTd.
#            define PYBIND11_BUILD_ABI "_mt_mscver" PYBIND11_PLATFORM_ABI_ID_TOSTRING(_MSC_VER)
#        else
#            if (_MSC_VER) / 100 == 19
#                define PYBIND11_BUILD_ABI "_none_mscver19"
#            else
#                error "Unknown major version for MSC_VER: PLEASE REVISE THIS CODE."
#            endif
#        endif
#    elif defined(__NVCOMPILER)       // NVHPC (PGI-based).
#        define PYBIND11_BUILD_ABI "" // TODO: What should be here, to prevent UB?
#    else
#        error "Unknown platform or compiler: PLEASE REVISE THIS CODE."
#    endif
#endif

#ifndef PYBIND11_INTERNALS_KIND
#    define PYBIND11_INTERNALS_KIND ""
#endif

#define PYBIND11_PLATFORM_ABI_ID                                                                  \
    PYBIND11_INTERNALS_KIND PYBIND11_COMPILER_TYPE PYBIND11_STDLIB PYBIND11_BUILD_ABI             \
        PYBIND11_BUILD_TYPE
