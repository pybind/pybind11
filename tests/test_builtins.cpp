/*
    tests/test_builtins.cpp -- python builtin functions

    Copyright (c) 2018 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#endif

TEST_SUBMODULE(builtins, m) {
    m.def("is_callable", [](py::object o) { return py::callable(o); });
}
