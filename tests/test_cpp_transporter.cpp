// Copyright (c) 2024 The pybind Community.

#include "pybind11_tests.h"
#include "test_cpp_transporter_traveler_bindings.h"

#include <typeinfo>

namespace pybind11_tests {
namespace test_cpp_transporter {

TEST_SUBMODULE(cpp_transporter, m) {
    m.attr("PYBIND11_PLATFORM_ABI_ID") = PYBIND11_PLATFORM_ABI_ID;
    m.attr("cap_cpp_type_info_Traveler")
        = py::capsule(&typeid(Traveler), "const std::type_info *");
    m.attr("cap_cpp_type_info_int") = py::capsule(&typeid(int), "const std::type_info *");

    wrap_traveler(m);
}

} // namespace test_cpp_transporter
} // namespace pybind11_tests
