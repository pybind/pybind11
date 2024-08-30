#include "pybind11_tests.h"
#include "test_cpp_transporter_traveler_bindings.h"

namespace pybind11_tests {
namespace test_cpp_transporter {

TEST_SUBMODULE(cpp_transporter, m) {
    m.attr("PYBIND11_PLATFORM_ABI_ID") = PYBIND11_PLATFORM_ABI_ID;
    m.attr("typeid_Traveler_name") = typeid(Traveler).name();

    wrap_traveler(m);
}

} // namespace test_cpp_transporter
} // namespace pybind11_tests
