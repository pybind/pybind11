#include <pybind11/pybind11.h>

namespace py = pybind11;

/* Simple test module/test class to check that the referenced internals data of external pybind11
 * modules aren't preserved over a finalize/initialize.
 */

namespace {
// Compare unsafe_reset_internals_for_single_interpreter in
// test_subinterpreter.cpp.
void unsafe_reset_local_internals() {
    // NOTE: This code is NOT SAFE unless the caller guarantees no other threads are alive
    // NOTE: This code is tied to the precise implementation of the internals holder

    py::detail::get_local_internals_pp_manager().unref();
    py::detail::get_local_internals();
}
} // namespace

PYBIND11_MODULE(external_module,
                m,
                py::mod_gil_not_used(),
                py::multiple_interpreters::per_interpreter_gil()) {
    // At least one test ("Single Subinterpreter") wants to reset
    // internals. We have separate local internals because we are a
    // separate DSO, so ours need to be reset too!
    unsafe_reset_local_internals();
    class A {
    public:
        explicit A(int value) : v{value} {};
        int v;
    };

    py::class_<A>(m, "A").def(py::init<int>()).def_readwrite("value", &A::v);

    m.def("internals_at",
          []() { return reinterpret_cast<uintptr_t>(&py::detail::get_internals()); });
}
