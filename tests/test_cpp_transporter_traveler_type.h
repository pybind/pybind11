#pragma once

#include <string>

namespace pybind11_tests {
namespace test_cpp_transporter {

struct Traveler {
    explicit Traveler(const std::string &luggage) : luggage(luggage) {}
    std::string luggage;
};

} // namespace test_cpp_transporter
} // namespace pybind11_tests
