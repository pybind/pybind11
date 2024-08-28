#pragma once

#include <string>

namespace pybind11_tests {
namespace test_cpp_transporter {

struct Traveler {
    explicit Traveler(const std::string &luggage) : luggage(luggage) {}
    std::string luggage;
};

struct PremiumTraveler : Traveler {
    explicit PremiumTraveler(const std::string &luggage, int points)
        : Traveler(luggage), points(points) {}
    int points;
};

} // namespace test_cpp_transporter
} // namespace pybind11_tests
