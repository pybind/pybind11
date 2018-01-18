/*
    tests/test_stl.cpp -- STL utility support

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include <pybind11/utility.h>

// Test with `std::variant` in C++17 mode, or with `boost::variant` in C++11/14
#if PYBIND11_HAS_VARIANT
using std::variant;
#elif defined(PYBIND11_TEST_BOOST) && (!defined(_MSC_VER) || _MSC_VER >= 1910)
#  include <boost/variant.hpp>
#  define PYBIND11_HAS_VARIANT 1
using boost::variant;

namespace pybind11 { namespace detail {
template <typename... Ts>
struct type_caster<boost::variant<Ts...>> : variant_caster<boost::variant<Ts...>> {};

template <>
struct visit_helper<boost::variant> {
    template <typename... Args>
    static auto call(Args &&...args) -> decltype(boost::apply_visitor(args...)) {
        return boost::apply_visitor(args...);
    }
};
}} // namespace pybind11::detail
#endif

/// Issue #528: templated constructor
struct TplCtorClass {
    template <typename T> TplCtorClass(const T &) { }
    bool operator==(const TplCtorClass &) const { return true; }
};

namespace std {
    template <>
    struct hash<TplCtorClass> { size_t operator()(const TplCtorClass &) const { return 0; } };
}


TEST_SUBMODULE(utility, m) {
    // Class that can be move- and copy-constructed, but not assigned
    struct NoAssign {
        int value;

        explicit NoAssign(int value = 0) : value(value) { }
        NoAssign(const NoAssign &) = default;
        NoAssign(NoAssign &&) = default;

        NoAssign &operator=(const NoAssign &) = delete;
        NoAssign &operator=(NoAssign &&) = delete;
    };
    py::class_<NoAssign>(m, "NoAssign", "Class with no C++ assignment operators")
        .def(py::init<>())
        .def(py::init<int>());

#ifdef PYBIND11_HAS_OPTIONAL
    // test_optional
    m.attr("has_optional") = true;

    using opt_int = std::optional<int>;
    using opt_no_assign = std::optional<NoAssign>;
    m.def("double_or_zero", [](const opt_int& x) -> int {
        return x.value_or(0) * 2;
    });
    m.def("half_or_none", [](int x) -> opt_int {
        return x ? opt_int(x / 2) : opt_int();
    });
    m.def("test_nullopt", [](opt_int x) {
        return x.value_or(42);
    }, py::arg_v("x", std::nullopt, "None"));
    m.def("test_no_assign", [](const opt_no_assign &x) {
        return x ? x->value : 42;
    }, py::arg_v("x", std::nullopt, "None"));

    m.def("nodefer_none_optional", [](std::optional<int>) { return true; });
    m.def("nodefer_none_optional", [](py::none) { return false; });
#endif

#ifdef PYBIND11_HAS_EXP_OPTIONAL
    // test_exp_optional
    m.attr("has_exp_optional") = true;

    using exp_opt_int = std::experimental::optional<int>;
    using exp_opt_no_assign = std::experimental::optional<NoAssign>;
    m.def("double_or_zero_exp", [](const exp_opt_int& x) -> int {
        return x.value_or(0) * 2;
    });
    m.def("half_or_none_exp", [](int x) -> exp_opt_int {
        return x ? exp_opt_int(x / 2) : exp_opt_int();
    });
    m.def("test_nullopt_exp", [](exp_opt_int x) {
        return x.value_or(42);
    }, py::arg_v("x", std::experimental::nullopt, "None"));
    m.def("test_no_assign_exp", [](const exp_opt_no_assign &x) {
        return x ? x->value : 42;
    }, py::arg_v("x", std::experimental::nullopt, "None"));
#endif

#ifdef PYBIND11_HAS_VARIANT
    static_assert(std::is_same<py::detail::variant_caster_visitor::result_type, py::handle>::value,
                  "visitor::result_type is required by boost::variant in C++11 mode");

    struct visitor {
        using result_type = const char *;

        result_type operator()(int) { return "int"; }
        result_type operator()(std::string) { return "std::string"; }
        result_type operator()(double) { return "double"; }
        result_type operator()(std::nullptr_t) { return "std::nullptr_t"; }
    };

    // test_variant
    m.def("load_variant", [](variant<int, std::string, double, std::nullptr_t> v) {
        return py::detail::visit_helper<variant>::call(visitor(), v);
    });
    m.def("load_variant_2pass", [](variant<double, int> v) {
        return py::detail::visit_helper<variant>::call(visitor(), v);
    });
    m.def("cast_variant", []() {
        using V = variant<int, std::string>;
        return py::make_tuple(V(5), V("Hello"));
    });
#endif

    // #528: templated constructor
    // (no python tests: the test here is that this compiles)
#if defined(PYBIND11_HAS_OPTIONAL)
    m.def("tpl_constr_optional", [](std::optional<TplCtorClass> &) {});
#elif defined(PYBIND11_HAS_EXP_OPTIONAL)
    m.def("tpl_constr_optional", [](std::experimental::optional<TplCtorClass> &) {});
#endif
}
