#include <pybind11/pybind11.h>

namespace py = pybind11;

/* Simple test module/test class to check that the referenced internals data of external pybind11
 * modules are different across subinterpreters
 */

PYBIND11_MODULE(mod_per_interpreter_gil,
                m,
                py::mod_gil_not_used(),
                py::multiple_interpreters::per_interpreter_gil()) {
    m.def("internals_at",
          []() { return reinterpret_cast<uintptr_t>(&py::detail::get_internals()); });
#ifdef PYBIND11_HAS_SUBINTERPRETER_SUPPORT
    m.attr("defined_PYBIND11_HAS_SUBINTERPRETER_SUPPORT") = true;
#else
    m.attr("defined_PYBIND11_HAS_SUBINTERPRETER_SUPPORT") = false;
#endif
}
