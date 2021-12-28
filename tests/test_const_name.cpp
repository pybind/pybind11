// Copyright (c) 2021 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pybind11_tests.h"

// IUT = Implementation Under Test
#define CONST_NAME_TESTS(TEST_FUNC, IUT)                                                          \
    std::string TEST_FUNC(int selector) {                                                         \
        switch (selector) {                                                                       \
            case 0:                                                                               \
                return IUT("").text;                                                              \
            case 1:                                                                               \
                return IUT("A").text;                                                             \
            case 2:                                                                               \
                return IUT("Bd").text;                                                            \
            case 3:                                                                               \
                return IUT("Cef").text;                                                           \
            case 4:                                                                               \
                return IUT<int>().text;                                                           \
            case 5:                                                                               \
                return IUT<std::string>().text;                                                   \
            case 6:                                                                               \
                return IUT<0>().text;                                                             \
            case 7:                                                                               \
                return IUT<1>().text;                                                             \
            case 8:                                                                               \
                return IUT<2>().text;                                                             \
            case 9:                                                                               \
                return IUT<34>().text;                                                            \
            case 10:                                                                              \
                return IUT<567>().text;                                                           \
            case 11:                                                                              \
                return IUT<true>("T1", "T2").text;                                                \
            case 12:                                                                              \
                return IUT<false>("U1", "U2").text;                                               \
            case 13:                                                                              \
                return IUT<true>(IUT("D1"), IUT("D2")).text;                                      \
            case 14:                                                                              \
                return IUT<false>(IUT("E1"), IUT("E2")).text;                                     \
            case 15:                                                                              \
                return IUT("KeepAtEnd").text;                                                     \
            default:                                                                              \
                break;                                                                            \
        }                                                                                         \
        throw std::runtime_error("Invalid selector value.");                                      \
    }

CONST_NAME_TESTS(const_name_tests, py::detail::const_name)

#ifdef PYBIND11_DETAIL_UNDERSCORE_BACKWARD_COMPATIBILITY
CONST_NAME_TESTS(underscore_tests, py::detail::_)
#endif

TEST_SUBMODULE(const_name, m) {
    m.def("const_name_tests", const_name_tests);

#ifdef PYBIND11_DETAIL_UNDERSCORE_BACKWARD_COMPATIBILITY
    m.def("underscore_tests", underscore_tests);
#else
    m.attr("underscore_tests") = py::none();
#endif
}
