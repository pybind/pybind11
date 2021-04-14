#include "pybind11_tests.h"

// The main purpose of this test is to ensure the suggested BOILERPLATE code block below is
// correct.

// Copy this block of code into your project.
// Replace FOOEXT with the name of your project.
// BOILERPLATE BEGIN
#ifdef FOOEXT_USING_PYBIND11_SMART_HOLDER
#    include <pybind11/smart_holder.h>
#else
#    include <pybind11/pybind11.h>
PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
template <typename type_, typename... options>
using classh = class_<type_, options...>;
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
#    define PYBIND11_SMART_HOLDER_TYPE_CASTERS(...)
#    define PYBIND11_TYPE_CASTER_BASE_HOLDER(...)
#endif
// BOILERPLATE END

namespace {
struct Foo0 {};
struct Foo1 {};
struct Foo2 {};
} // namespace

PYBIND11_TYPE_CASTER_BASE_HOLDER(Foo1, std::shared_ptr<Foo1>)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(Foo2)

TEST_SUBMODULE(classh_mock, m) {
    // Uses std::unique_ptr<Foo0> as holder in conservative mode, py::smart_holder in progressive
    // mode (if available).
    py::class_<Foo0>(m, "Foo0").def(py::init<>());

    // Always uses std::shared_ptr<Foo1> as holder.
    py::class_<Foo1, std::shared_ptr<Foo1>>(m, "Foo1").def(py::init<>());

    // Uses py::smart_holder if available, or std::unique_ptr<Foo2> if only pybind11 classic is
    // available.
    py::classh<Foo2>(m, "Foo2").def(py::init<>());
}
