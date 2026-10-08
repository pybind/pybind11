/*
    tests/test_chrono_second_tu.cpp -- chrono casters from a second translation unit

    Only durations are used here. If the datetime helpers have external linkage, the linker can
    pair this TU's `datetime_import()` with test_chrono.cpp's `is_datetime()`, which then reads
    a null `PyDateTimeAPI`.

    Copyright (c) 2026 Henry Schreiner

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <pybind11/chrono.h>

#include "pybind11_tests.h"

#include <chrono>

TEST_SUBMODULE(chrono_second_tu, m) {
    m.def("duration_roundtrip",
          [](const std::chrono::system_clock::duration &d) { return d; });
}
