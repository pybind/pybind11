#pragma once

#include <pybind11/pybind11.h>

#include "test_cpp_transporter_traveler_types.h"

#include <cstring>
#include <string>

PYBIND11_NAMESPACE_BEGIN(pybind11)

inline object cpp_transporter(handle self,
                              const str &pybind11_platform_abi_id,
                              const capsule &cap_cpp_type_info,
                              const str &pointer_kind) {
    auto pointer_kind_cpp = pointer_kind.cast<std::string>();
    if (pybind11_platform_abi_id.cast<std::string>() != PYBIND11_PLATFORM_ABI_ID) {
        if (pointer_kind_cpp == "query_mismatch") {
            return cast("pybind11_platform_abi_id_mismatch");
        }
        return none();
    }
    if (std::strcmp(cap_cpp_type_info.name(), "const std::type_info *") != 0) {
        if (pointer_kind_cpp == "query_mismatch") {
            return cast("cap_cpp_type_info_name_mismatch");
        }
        return none();
    }
    if (pointer_kind_cpp != "raw_pointer_ephemeral") {
        throw std::runtime_error("Unknown pointer_kind: \"" + pointer_kind_cpp + "\"");
    }
    const auto *cpp_type_info = cap_cpp_type_info.get_pointer<const std::type_info>();
    detail::type_caster_generic caster(*cpp_type_info);
    if (!caster.load(self, false)) {
        return none();
    }
    return capsule(caster.value, cpp_type_info->name());
}

PYBIND11_NAMESPACE_END(pybind11)

namespace pybind11_tests {
namespace test_cpp_transporter {

namespace py = pybind11;

inline void wrap_traveler(py::module_ m) {
    m.attr("PYBIND11_PLATFORM_ABI_ID") = PYBIND11_PLATFORM_ABI_ID;
    m.attr("typeid_Traveler_name") = typeid(Traveler).name();

    py::class_<Traveler>(m, "Traveler")
        .def("__cpp_transporter__", py::cpp_transporter)
        .def(py::init<std::string>())
        .def_readwrite("luggage", &Traveler::luggage);

    m.def("get_luggage", [](const Traveler &person) { return person.luggage; });

    py::class_<PremiumTraveler, Traveler>(m, "PremiumTraveler")
        .def(py::init<std::string, int>())
        .def_readwrite("points", &PremiumTraveler::points);

    m.def("get_points", [](const PremiumTraveler &person) { return person.points; });
}

} // namespace test_cpp_transporter
} // namespace pybind11_tests
