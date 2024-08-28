#pragma once

#include <pybind11/pybind11.h>

#include "test_cpp_transporter_traveler_types.h"

#include <cstring>
#include <string>

namespace pybind11_tests {
namespace test_cpp_transporter {

namespace py = pybind11;

inline void wrap_traveler(py::module_ m) {
    m.attr("PYBIND11_PLATFORM_ABI_ID") = PYBIND11_PLATFORM_ABI_ID;
    m.attr("typeid_Traveler_name") = typeid(Traveler).name();

    py::class_<Traveler>(m, "Traveler")
        .def(py::init<std::string>())
        .def("__cpp_transporter__",
             [](py::handle self,
                const py::str &pybind11_platform_abi_id,
                const py::str &cpp_typeid_name,
                const py::str &pointer_kind) -> py::object {
                 auto pointer_kind_cpp = pointer_kind.cast<std::string>();
                 if (pybind11_platform_abi_id.cast<std::string>() != PYBIND11_PLATFORM_ABI_ID) {
                     if (pointer_kind_cpp == "query_mismatch") {
                         return py::cast("pybind11_platform_abi_id_mismatch");
                     }
                     return py::none();
                 }
                 if (cpp_typeid_name.cast<std::string>() != typeid(Traveler).name()) {
                     if (pointer_kind_cpp == "query_mismatch") {
                         return py::cast("cpp_typeid_name_mismatch");
                     }
                     return py::none();
                 }
                 if (pointer_kind_cpp != "raw_pointer_ephemeral") {
                     throw std::runtime_error("Unknown pointer_kind: \"" + pointer_kind_cpp
                                              + "\"");
                 }
                 auto *self_cpp_ptr = py::cast<Traveler *>(self);
                 return py::capsule(static_cast<void *>(self_cpp_ptr), typeid(Traveler).name());
             })
        .def_readwrite("luggage", &Traveler::luggage);

    m.def("get_luggage", [](const Traveler &person) { return person.luggage; });

    py::class_<PremiumTraveler, Traveler>(m, "PremiumTraveler")
        .def(py::init<std::string, int>())
        .def_readwrite("points", &PremiumTraveler::points);

    m.def("get_points", [](const PremiumTraveler &person) { return person.points; });
}

} // namespace test_cpp_transporter
} // namespace pybind11_tests
