#include <pybind11/pybind11.h>

#include "pet.hpp"

#include <string>

namespace py = pybind11;

PYBIND11_MODULE(pet, m) {
    py::class_<Pet> pet(m, "Pet");
    pet.def(py::init<const std::string &>()).def_readwrite("name", &Pet::name);
}
