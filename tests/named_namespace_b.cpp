#include "pybind11_tests.h"

namespace test_named_namespace {
struct any_struct {};
} // namespace test_named_namespace

PYBIND11_MODULE(named_namespace_b, m) {
    m.attr("name") = "NB";

    py::detail::get_internals()
        .std_type_index_registry_named_namespace[std::type_index(
            typeid(test_named_namespace::any_struct))]
        .push_back("NB");
}
