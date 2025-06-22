/*
    tests/make_alias -- make alias

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <pybind11/pybind11.h>

#include "constructor_stats.h"
#include "pybind11_tests.h"

#if defined(_MSC_VER)
#    pragma warning(disable : 4996) // C4996: std::unary_negation is deprecated
#endif

TEST_SUBMODULE(_make_alias, m) {
    ma = m.make_alias(pybind11::module::strip_leading_underscore_from_name{});
    ma.def("foo", []() -> {});
}
