#pragma once

#include <pybind11/pybind11.h>

#include "test_cpp_transporter_traveler_type.h"

#include <string>

namespace pybind11_tests {
namespace test_cpp_transporter {

namespace py = pybind11;

inline void wrap_traveler(py::module_ m) {
    py::class_<Traveler>(m, "Traveler")
        .def(py::init<std::string>())
        .def("__cpp_transporter__",
             [](py::handle self,
                const py::str & /*cpp_abi_code*/,
                const py::str & /*cpp_typeid_name*/,
                const py::str &pointer_kind) {
                 auto pointer_kind_cpp = pointer_kind.cast<std::string>();
                 if (pointer_kind_cpp != "raw_pointer_ephemeral") {
                     throw std::runtime_error("Unknown pointer_kind: \"" + pointer_kind_cpp
                                              + "\"");
                 }
                 auto *self_cpp_ptr = py::cast<Traveler *>(self);
                 return py::capsule(static_cast<void *>(self_cpp_ptr), typeid(Traveler).name());
             })
        .def_readwrite("luggage", &Traveler::luggage);

    m.def("get_luggage", [](const Traveler &person) { return person.luggage; });
};

} // namespace test_cpp_transporter
} // namespace pybind11_tests
