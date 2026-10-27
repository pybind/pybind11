#include <pybind11/pybind11.h>

#include "dog.hpp"

#include <string>

namespace py = pybind11;

PYBIND11_MODULE(dog, m) {
    py::class_<Dog, Pet>(m, "Dog").def(py::init<const std::string &>()).def("bark", &Dog::bark);
}
