#include "pybind11_tests.h"

namespace adl_one {
struct type_one {};
struct type_caster_one {
    static int num() { return 101; }
};
type_caster_one pybind11_select_caster(type_one *);
} // namespace adl_one

namespace adl_two {
struct type_two {};
struct type_caster_two {
    static int num() { return 202; }
};
type_caster_two pybind11_select_caster(type_two *);
} // namespace adl_two

TEST_SUBMODULE(make_caster_adl, m) {
    m.def("num_one", []() { return py::detail::make_caster<adl_one::type_one>::num(); });
    m.def("num_two", []() { return py::detail::make_caster<adl_two::type_two>::num(); });
}
