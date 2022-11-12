// #include <pybind11/native_enum.h>

#include "pybind11_tests.h"

namespace test_native_enum {

// https://en.cppreference.com/w/cpp/language/enum

// enum that takes 16 bits
enum smallenum : std::int16_t { a, b, c };

// color may be red (value 0), yellow (value 1), green (value 20), or blue (value 21)
enum color { red, yellow, green = 20, blue };

// altitude may be altitude::high or altitude::low
enum class altitude : char {
    high = 'h',
    low = 'l', // trailing comma only allowed after CWG518
};

// the constant d is 0, the constant e is 1, the constant f is 3
enum { d, e, f = e + 2 };

int pass_color(color e) { return static_cast<int>(e); }
color return_color(int i) { return static_cast<color>(i); }

py::handle wrap_color(py::module_ m) {
    auto enum_module = py::module_::import("enum");
    auto int_enum = enum_module.attr("IntEnum");
    using u_t = std::underlying_type<color>::type;
    auto members = py::make_tuple(py::make_tuple("red", static_cast<u_t>(color::red)),
                                  py::make_tuple("yellow", static_cast<u_t>(color::yellow)),
                                  py::make_tuple("green", static_cast<u_t>(color::green)),
                                  py::make_tuple("blue", static_cast<u_t>(color::blue)));
    auto int_enum_color = int_enum("color", members);
    int_enum_color.attr("__module__") = m;
    m.attr("color") = int_enum_color;
    return int_enum_color.release(); // Intentionally leak Python reference.
}

} // namespace test_native_enum

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

using namespace test_native_enum;

template <>
struct type_caster<color> {
    static handle native_type;

    static handle cast(const color &src, return_value_policy /* policy */, handle /* parent */) {
        auto u_v = static_cast<std::underlying_type<color>::type>(src);
        return native_type(u_v).release();
    }

    bool load(handle src, bool /* convert */) {
        if (!isinstance(src, native_type)) {
            return false;
        }
        value = static_cast<color>(py::cast<std::underlying_type<color>::type>(src.attr("value")));
        return true;
    }

    PYBIND11_TYPE_CASTER(color, const_name("<enum 'color'>"));
};

handle type_caster<color>::native_type = nullptr;

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

TEST_SUBMODULE(native_enum, m) {
    using namespace test_native_enum;

    py::detail::type_caster<color>::native_type = wrap_color(m);

    m.def("pass_color", pass_color);
    m.def("return_color", return_color);
}
