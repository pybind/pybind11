#include "pybind11_tests.h"

namespace have_a_ns {
struct type_mock {};
struct mock_caster_alt {
    static int num() { return 121; }
};
mock_caster_alt pybind11_select_caster(type_mock *);
} // namespace have_a_ns

TEST_SUBMODULE(make_caster_adl_alt, m) {
    m.def("have_a_ns_num", []() { return py::detail::make_caster<have_a_ns::type_mock>::num(); });
}
