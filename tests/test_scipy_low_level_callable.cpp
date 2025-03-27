#include "pybind11_tests.h"

namespace pybind11_tests {
namespace scipy_low_level_callable {

extern "C" double square(double x) { return x * x; }

} // namespace scipy_low_level_callable
} // namespace pybind11_tests

TEST_SUBMODULE(scipy_low_level_callable, m) {
    using namespace pybind11_tests::scipy_low_level_callable;

    m.def("square", square);
}
