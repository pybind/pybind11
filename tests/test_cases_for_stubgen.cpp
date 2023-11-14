#include "pybind11/stl_bind.h"
#include "pybind11_tests.h"

#include <map>

namespace test_cases_for_stubgen {

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

TEST_SUBMODULE(cases_for_stubgen, m) {
    using UserType = test_cases_for_stubgen::UserType;

    m.def("pass_user_type", [](const UserType &) {});
    m.def("return_user_type", []() { return UserType(); });

    py::bind_map<std::map<int, UserType>>(m, "MapIntUserType");
    py::bind_map<std::map<UserType, int>>(m, "MapUserTypeInt");

#define MAP_TYPE(MapTypePythonName, ...)                                                          \
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

    MAP_TYPE("MapFloatUserType", std::map<float, UserType>);
    MAP_TYPE("MapUserTypeFloat", std::map<UserType, float>);

#undef MAP_TYPE
}
