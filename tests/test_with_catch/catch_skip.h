// Macro to skip a test at runtime with a visible message.
// Catch2 v2 doesn't have native skip support (v3 does with SKIP()).
// The test will count as "passed" in totals, but the output clearly shows it was skipped.

#pragma once

#include <catch.hpp>

#define PYBIND11_CATCH2_SKIP_IF(condition, reason)                                                \
    do {                                                                                          \
        if (condition) {                                                                          \
            Catch::cout() << "[ SKIPPED  ] " << (reason) << '\n';                                 \
            Catch::cout().flush();                                                                \
            return;                                                                               \
        }                                                                                         \
    } while (0)
