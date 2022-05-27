#include "pybind11_tests.h"

namespace test_perf_accessors {

py::object perf_list_accessor(std::size_t num_iterations, int test_id) {
    py::list l;
    l.append(0);
    auto var = l[0]; // Type of var is list accessor.
    py::int_ answer(42);
    var = answer; // Detach var from list, which releases a reference.
#ifdef PYBIND11_HANDLE_REF_DEBUG
    if (num_iterations == 0) {
        py::list return_list;
        std::size_t inc_refs = py::handle::inc_ref_counter();
        var = answer;
        inc_refs = py::handle::inc_ref_counter() - inc_refs;
        return_list.append(inc_refs);
        inc_refs = py::handle::inc_ref_counter();
        var = 42;
        inc_refs = py::handle::inc_ref_counter() - inc_refs;
        return_list.append(inc_refs);
        return return_list;
    }
#endif
    if (test_id == 0) {
        while (num_iterations != 0u) {
            var = answer;
            num_iterations--;
        }
    } else {
        while (num_iterations != 0u) {
            var = 42;
            num_iterations--;
        }
    }
    return py::none();
}

} // namespace test_perf_accessors

TEST_SUBMODULE(perf_accessors, m) {
    using namespace test_perf_accessors;
    m.def("perf_list_accessor", perf_list_accessor);
}
