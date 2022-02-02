#include "pybind11_tests.h"

namespace {
struct any_struct {};
} // namespace

TEST_SUBMODULE(unnamed_namespace_b, m) {
    m.attr("name") = "B";

#if defined(PYBIND11_TEST_BOOST)
    py::detail::get_internals()
        .boost_type_index_registry[boost::typeindex::type_index(
            boost::typeindex::type_id<any_struct>())]
        .push_back("B");
#endif
}
