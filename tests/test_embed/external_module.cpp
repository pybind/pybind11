#include <pybind11/pybind11.h>

namespace py = pybind11;

/* Simple test module/test class to check that the referenced internals data of external pybind11
 * modules aren't preserved over a finalize/initialize.
 */

PYBIND11_MODULE(external_module, m) {
    class A {
    public:
        explicit A(int value) : v{value} {};
        int v;
    };

    class B {};

    py::class_<A>(m, "A").def(py::init<int>()).def_readwrite("value", &A::v);

    // PR #3798 : Local internals must be cleared on finalization so they can be registered again.
    // This class is not explicitly used in the test, but it is still registered / unregistered
    // when the interpreter stops and restarts.
    py::class_<B>(m, "B", py::module_local());

    m.def("internals_at",
          []() { return reinterpret_cast<uintptr_t>(&py::detail::get_internals()); });
}
