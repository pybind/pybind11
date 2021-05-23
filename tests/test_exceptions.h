#pragma once
#include "pybind11_tests.h"
#include <stdexcept>

// shared exceptions for cross_module_tests

class tmp_e : public std::runtime_error {
public:
    explicit tmp_e() : std::runtime_error("") {}
};
