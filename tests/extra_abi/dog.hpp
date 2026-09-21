#include <pybind11/pybind11.h>

#include "pet.hpp"

#include <string>

struct PYBIND11_EXPORT Dog : Pet {
    explicit Dog(const std::string &name) : Pet(name) {}
    std::string bark() const { return "woof!"; }
};
