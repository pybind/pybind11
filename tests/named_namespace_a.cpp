#include <pybind11/stl.h>

#include "pybind11_tests.h"

namespace test_named_namespace {
struct any_struct {};
} // namespace test_named_namespace

PYBIND11_MODULE(named_namespace_a, m) {
    m.attr("name") = "NA";

    py::detail::get_internals()
        .std_type_index_registry_named_namespace[std::type_index(
            typeid(test_named_namespace::any_struct))]
        .push_back("NA");

    m.def("std_type_index_registry_dump", []() {
        py::list items;
        for (const auto &it :
             py::detail::get_internals().std_type_index_registry_named_namespace) {
            items.append(py::make_tuple(it.first.name(), it.second));
        }
        return items;
    });
}
