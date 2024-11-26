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
#    if defined(__MINGW32__)
#        define PYBIND11_COMPILER_TYPE "_mingw"
#    elif defined(__CYGWIN__)
#        define PYBIND11_COMPILER_TYPE "_gcc_cygwin"
#    elif defined(_MSC_VER)
#        define PYBIND11_COMPILER_TYPE "_msvc"
#    elif defined(__PGI)
#        define PYBIND11_COMPILER_TYPE "_pgi"
#    elif defined(__INTEL_COMPILER) || defined(__clang__) || defined(__GNUC__)
#        define PYBIND11_COMPILER_TYPE "_system" // Assumed compatible with system compiler.
#    else
#        error "Unknown PYBIND11_COMPILER_TYPE: PLEASE REVISE THIS CODE."
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
#    if defined(_MSC_VER)                 // See PR #4953.
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
#    elif defined(__NVCOMPILER)        // NVHPC (PGI-based).
#        define PYBIND11_BUILD_ABI ""  // TODO: What should be here, to prevent UB?
#    elif defined(_LIBCPP_ABI_VERSION) // https://libcxx.llvm.org/DesignDocs/ABIVersioning.html
#        define PYBIND11_BUILD_ABI "_abi" PYBIND11_PLATFORM_ABI_ID_TOSTRING(_LIBCPP_ABI_VERSION)
#    elif defined(__GXX_ABI_VERSION)
#        if __GXX_ABI_VERSION >= 1002 && __GXX_ABI_VERSION < 2000
#            if !defined(_GLIBCXX_USE_CXX11_ABI)
#                error "UNEXPECTED: _GLIBCXX_USE_CXX11_ABI not defined: PLEASE REVISE THIS CODE."
#            endif
#            define PYBIND11_BUILD_ABI                                                            \
                "_gxx_abi_1xxx_use_cxx11_abi_" PYBIND11_PLATFORM_ABI_ID_TOSTRING(                 \
                    _GLIBCXX_USE_CXX11_ABI)
#        else
#            error "Unknown platform or compiler (__GXX_ABI_VERSION): PLEASE REVISE THIS CODE."
#        endif
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
