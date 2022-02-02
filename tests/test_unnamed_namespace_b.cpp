#include "pybind11_tests.h"

namespace {
struct any_struct {};
} // namespace

TEST_SUBMODULE(unnamed_namespace_b, m) {
    m.attr("name") = "B";

    py::detail::get_internals()
        .std_type_index_registry[std::type_index(typeid(any_struct))]
        .push_back("B");
}
