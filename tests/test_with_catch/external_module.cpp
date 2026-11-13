#include <pybind11/pybind11.h>

namespace py = pybind11;

/* Simple test module/test class to check that the referenced internals data of external pybind11
 * modules aren't preserved over a finalize/initialize.
 */

PYBIND11_MODULE(external_module,
                m,
                py::mod_gil_not_used(),
                py::multiple_interpreters::per_interpreter_gil()) {
    // A separate DSO must refresh its own local cache after an interpreter restart. Check before
    // registering bindings, which would dereference a stale holder if initialization missed it.
    auto *local_internals_pp = py::detail::get_local_internals_pp_manager().get_pp();
    auto *local_internals_capsule = py::detail::get_local_internals_capsule();
    if (!local_internals_capsule
        || PyCapsule_GetPointer(local_internals_capsule, nullptr) != local_internals_pp) {
        throw py::import_error("external_module has a stale local internals cache");
    }
    class A {
    public:
        explicit A(int value) : v{value} {};
        int v;
    };

    py::class_<A>(m, "A").def(py::init<int>()).def_readwrite("value", &A::v);

    m.def("internals_at",
          []() { return reinterpret_cast<uintptr_t>(&py::detail::get_internals()); });
}
