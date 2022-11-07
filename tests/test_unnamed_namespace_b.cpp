#include "pybind11_tests.h"

namespace {
struct any_struct {};
} // namespace

TEST_SUBMODULE(unnamed_namespace_b, m) {
    m.attr("name") = "UB";

    py::detail::get_internals()
        .std_type_index_registry_unnamed_namespace[std::type_index(typeid(any_struct))]
        .push_back("UB");
}
