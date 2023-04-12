#include "pybind11_tests.h"

namespace pybind11_tests {
namespace blank_page {

struct OptionsBase {
    int simple_value() const { return simple_value_; }

    OptionsBase &SetSimpleValue(int simple_value) {
        simple_value_ = simple_value;
        return *this;
    }

private:
    int simple_value_ = -99;
};

struct Options : OptionsBase {};

} // namespace blank_page
} // namespace pybind11_tests

TEST_SUBMODULE(blank_page, m) {
    using namespace pybind11_tests::blank_page;

    py::class_<OptionsBase>(m, "OptionsBase");

    py::class_<Options>(m, "Options")
        .def(py::init<>())
        .def_property("simple_value", &Options::simple_value, &Options::SetSimpleValue);
}
