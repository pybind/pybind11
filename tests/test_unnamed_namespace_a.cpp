#include <pybind11/stl.h>

#include "pybind11_tests.h"

namespace {
struct any_struct {};
} // namespace

TEST_SUBMODULE(unnamed_namespace_a, m) {
    m.attr("name") = "A";

    py::detail::get_internals()
        .std_type_index_registry[std::type_index(typeid(any_struct))]
        .push_back("A");

    m.def("std_type_index_registry_dump", []() {
        py::list items;
        for (const auto &it : py::detail::get_internals().std_type_index_registry) {
            items.append(py::make_tuple(it.first.name(), it.second));
        }
        return items;
    });
}
