// Macro to skip a test at runtime with a visible message.
// Catch2 v2 doesn't have native skip support (v3 does with SKIP()).
// The test will count as "passed" in totals, but the output clearly shows it was skipped.

#pragma once

#include <pybind11/detail/pybind11_namespace_macros.h>

#include <catch.hpp>

#define PYBIND11_CATCH2_SKIP_IF(condition, reason)                                                \
    do {                                                                                          \
        PYBIND11_WARNING_PUSH                                                                     \
        PYBIND11_WARNING_DISABLE_MSVC(4127)                                                       \
        if (condition) {                                                                          \
            Catch::cout() << "[ SKIPPED  ] " << (reason) << '\n';                                 \
            Catch::cout().flush();                                                                \
            return;                                                                               \
        }                                                                                         \
        PYBIND11_WARNING_POP                                                                      \
    } while (0)
