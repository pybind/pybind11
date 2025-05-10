#include <pybind11/pybind11.h>

namespace py = pybind11;

/* Simple test module/test class to check that the referenced internals data of external pybind11
 * modules are different across subinterpreters
 */

PYBIND11_MODULE(mod_test_interpreters2,
                m,
                py::multiple_interpreters(py::multiple_interpreters::shared_gil)) {
    m.def("internals_at",
          []() { return reinterpret_cast<uintptr_t>(&py::detail::get_internals()); });
}
