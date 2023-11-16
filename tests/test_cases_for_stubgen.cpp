#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"
#include "pybind11_tests.h"

#include <array>
#include <map>
#include <vector>

namespace test_cases_for_stubgen {

// The `basics` code was copied from here (to have all test cases for stubgen in one place):
// https://github.com/python/mypy/blob/c6cb3c6282003dd3dadcf028735f9ba6190a0c84/test-data/pybind11_mypy_demo/src/main.cpp
// Copyright (c) 2016 The Pybind Development Team, All rights reserved.

namespace basics {

int answer() { return 42; }

int sum(int a, int b) { return a + b; }

double midpoint(double left, double right) { return left + (right - left) / 2; }

double weighted_midpoint(double left, double right, double alpha = 0.5) {
    return left + (right - left) * alpha;
}

struct Point {

    enum class LengthUnit { mm = 0, pixel, inch };

    enum class AngleUnit { radian = 0, degree };

    Point() : Point(0, 0) {}
    Point(double x, double y) : x(x), y(y) {}

    static const Point origin;
    static const Point x_axis;
    static const Point y_axis;

    static LengthUnit length_unit;
    static AngleUnit angle_unit;

    double length() const { return std::sqrt(x * x + y * y); }

    double distance_to(double other_x, double other_y) const {
        double dx = x - other_x;
        double dy = y - other_y;
        return std::sqrt(dx * dx + dy * dy);
    }

    double distance_to(const Point &other) const { return distance_to(other.x, other.y); }

    double x, y;
};

const Point Point::origin = Point(0, 0);
const Point Point::x_axis = Point(1, 0);
const Point Point::y_axis = Point(0, 1);

Point::LengthUnit Point::length_unit = Point::LengthUnit::mm;
Point::AngleUnit Point::angle_unit = Point::AngleUnit::radian;

} // namespace basics

void bind_basics(py::module &basics) {

    using namespace basics;

    // Functions
    basics.def(
        "answer", &answer, "answer docstring, with end quote\""); // tests explicit docstrings
    basics.def("sum", &sum, "multiline docstring test, edge case quotes \"\"\"'''");
    basics.def("midpoint", &midpoint, py::arg("left"), py::arg("right"));
    basics.def("weighted_midpoint",
               weighted_midpoint,
               py::arg("left"),
               py::arg("right"),
               py::arg("alpha") = 0.5);

    // Classes
    py::class_<Point> pyPoint(basics, "Point");
    py::enum_<Point::LengthUnit> pyLengthUnit(pyPoint, "LengthUnit");
    py::enum_<Point::AngleUnit> pyAngleUnit(pyPoint, "AngleUnit");

    pyPoint.def(py::init<>())
        .def(py::init<double, double>(), py::arg("x"), py::arg("y"))
#ifdef PYBIND11_CPP14
        .def("distance_to",
             py::overload_cast<double, double>(&Point::distance_to, py::const_),
             py::arg("x"),
             py::arg("y"))
        .def("distance_to",
             py::overload_cast<const Point &>(&Point::distance_to, py::const_),
             py::arg("other"))
#else
        .def("distance_to",
             static_cast<double (Point::*)(double, double) const>(&Point::distance_to),
             py::arg("x"),
             py::arg("y"))
        .def("distance_to",
             static_cast<double (Point::*)(const Point &) const>(&Point::distance_to),
             py::arg("other"))
#endif
        .def_readwrite("x", &Point::x)
        .def_property(
            "y",
            [](Point &self) { return self.y; },
            [](Point &self, double value) { self.y = value; })
        .def_property_readonly("length", &Point::length)
        .def_property_readonly_static("x_axis", [](py::handle /*cls*/) { return Point::x_axis; })
        .def_property_readonly_static("y_axis", [](py::handle /*cls*/) { return Point::y_axis; })
        .def_readwrite_static("length_unit", &Point::length_unit)
        .def_property_static(
            "angle_unit",
            [](py::handle /*cls*/) { return Point::angle_unit; },
            [](py::handle /*cls*/, Point::AngleUnit value) { Point::angle_unit = value; });

    pyPoint.attr("origin") = Point::origin;

    pyLengthUnit.value("mm", Point::LengthUnit::mm)
        .value("pixel", Point::LengthUnit::pixel)
        .value("inch", Point::LengthUnit::inch);

    pyAngleUnit.value("radian", Point::AngleUnit::radian)
        .value("degree", Point::AngleUnit::degree);

    // Module-level attributes
    basics.attr("PI") = std::acos(-1);
    basics.attr("__version__") = "0.0.1";
}

struct UserType {
    bool operator<(const UserType &) const { return false; }
};

struct minimal_caster {
    static constexpr auto name = py::detail::const_name<UserType>();

    static py::handle
    cast(UserType const & /*src*/, py::return_value_policy /*policy*/, py::handle /*parent*/) {
        return py::none().release();
    }

    // Maximizing simplicity. This will go terribly wrong for other arg types.
    template <typename>
    using cast_op_type = const UserType &;

    // NOLINTNEXTLINE(google-explicit-constructor)
    operator UserType const &() {
        static UserType obj;
        return obj;
    }

    bool load(py::handle /*src*/, bool /*convert*/) { return false; }
};

} // namespace test_cases_for_stubgen

namespace pybind11 {
namespace detail {

template <>
struct type_caster<test_cases_for_stubgen::UserType> : test_cases_for_stubgen::minimal_caster {};

} // namespace detail
} // namespace pybind11

PYBIND11_MAKE_OPAQUE(std::map<int, test_cases_for_stubgen::UserType>);
PYBIND11_MAKE_OPAQUE(std::map<test_cases_for_stubgen::UserType, int>);
PYBIND11_MAKE_OPAQUE(std::map<float, test_cases_for_stubgen::UserType>);
PYBIND11_MAKE_OPAQUE(std::map<test_cases_for_stubgen::UserType, float>);

TEST_SUBMODULE(cases_for_stubgen, m) {
    auto basics = m.def_submodule("basics");
    test_cases_for_stubgen::bind_basics(basics);

    using UserType = test_cases_for_stubgen::UserType;

    m.def("pass_user_type", [](const UserType &) {});
    m.def("return_user_type", []() { return UserType(); });

    py::bind_map<std::map<int, UserType>>(m, "MapIntUserType");
    py::bind_map<std::map<UserType, int>>(m, "MapUserTypeInt");

#define LOCAL_HELPER(MapTypePythonName, ...)                                                      \
    py::class_<__VA_ARGS__>(m, MapTypePythonName)                                                 \
        .def(                                                                                     \
            "keys",                                                                               \
            [](const __VA_ARGS__ &v) { return py::make_key_iterator(v); },                        \
            py::keep_alive<0, 1>())                                                               \
        .def(                                                                                     \
            "values",                                                                             \
            [](const __VA_ARGS__ &v) { return py::make_value_iterator(v); },                      \
            py::keep_alive<0, 1>())                                                               \
        .def(                                                                                     \
            "__iter__",                                                                           \
            [](const __VA_ARGS__ &v) { return py::make_iterator(v.begin(), v.end()); },           \
            py::keep_alive<0, 1>())

    LOCAL_HELPER("MapFloatUserType", std::map<float, UserType>);
    LOCAL_HELPER("MapUserTypeFloat", std::map<UserType, float>);
#undef LOCAL_HELPER

    m.def("pass_std_array_int_2", [](const std::array<int, 2> &) {});
    m.def("return_std_array_int_3", []() { return std::array<int, 3>{{1, 2, 3}}; });

    // Rather arbitrary, meant to be a torture test for recursive processing.
    using nested_case_01a = std::vector<std::array<int, 2>>;
    using nested_case_02a = std::vector<UserType>;
    using nested_case_03a = std::map<std::array<int, 2>, UserType>;
    using nested_case_04a = std::map<nested_case_01a, nested_case_02a>;
    using nested_case_05a = std::vector<nested_case_04a>;
    using nested_case_06a = std::map<nested_case_04a, nested_case_05a>;
#define LOCAL_HELPER(name) m.def(#name, [](const name &) {})
    LOCAL_HELPER(nested_case_01a);
    LOCAL_HELPER(nested_case_02a);
    LOCAL_HELPER(nested_case_03a);
    LOCAL_HELPER(nested_case_04a);
    LOCAL_HELPER(nested_case_05a);
    LOCAL_HELPER(nested_case_06a);
#undef LOCAL_HELPER
}
