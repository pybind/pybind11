#include <pybind11/native_enum.h>

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

} // namespace test_native_enum

TEST_SUBMODULE(native_enum, m) {
    using namespace test_native_enum;

    py::native_enum<color>(m, "color")
        .value("red", color::red)
        .value("yellow", color::yellow)
        .value("green", color::green)
        .value("blue", color::blue);

    m.def("pass_color", pass_color);
    m.def("return_color", return_color);
}
