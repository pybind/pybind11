#include <pybind11/stl.h>

#include "pybind11_tests.h"

namespace {
struct any_struct {};
} // namespace

TEST_SUBMODULE(unnamed_namespace_a, m) {
    m.attr("name") = "A";

#if defined(PYBIND11_TEST_BOOST)
    py::detail::get_internals()
        .boost_type_index_registry[boost::typeindex::type_index(
            boost::typeindex::type_id<any_struct>())]
        .push_back("A");
#endif

    m.def("boost_type_index_registry_dump", []() {
#if defined(PYBIND11_TEST_BOOST)
        py::list items;
        for (const auto &it : py::detail::get_internals().boost_type_index_registry) {
            items.append(py::make_tuple(it.first.pretty_name(), it.second));
        }
        return items;
#else
        return py::none();
#endif
    });
}
