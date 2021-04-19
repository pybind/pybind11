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
#    ifndef PYBIND11_SH_AVL
#        define PYBIND11_SH_AVL(...) std::shared_ptr<__VA_ARGS__> // "Smart_Holder if AVaiLable"
#    endif
#    ifndef PYBIND11_SH_DEF
#        define PYBIND11_SH_DEF(...) std::shared_ptr<__VA_ARGS__> // "Smart_Holder if DEFault"
#    endif
#    ifndef PYBIND11_SMART_HOLDER_TYPE_CASTERS
#        define PYBIND11_SMART_HOLDER_TYPE_CASTERS(...)
#    endif
#    ifndef PYBIND11_TYPE_CASTER_BASE_HOLDER
#        define PYBIND11_TYPE_CASTER_BASE_HOLDER(...)
#    endif
#endif
// BOILERPLATE END

namespace {
struct Foo0 {};
struct Foo1 {};
struct Foo2 {};
struct Foo3 {};
struct Foo4 {};
} // namespace

PYBIND11_TYPE_CASTER_BASE_HOLDER(Foo1, std::shared_ptr<Foo1>)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(Foo2)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(Foo4)

TEST_SUBMODULE(classh_mock, m) {
    // Please see README_smart_holder.rst, in particular section
    // Classic / Conservative / Progressive cross-module compatibility

    // Uses std::unique_ptr<Foo0> as holder in Conservative mode, py::smart_holder in Progressive
    // mode (if available).
    py::class_<Foo0>(m, "Foo0").def(py::init<>());

    // Always uses std::shared_ptr<Foo1> as holder.
    py::class_<Foo1, std::shared_ptr<Foo1>>(m, "Foo1").def(py::init<>());

    // Uses std::shared_ptr<Foo2> as holder in Classic mode, py::smart_holder in Conservative or
    // Progressive mode.
    py::class_<Foo2, PYBIND11_SH_AVL(Foo2)>(m, "Foo2").def(py::init<>());
    // ------------- std::shared_ptr<Foo2> -- same length by design, to not disturb the indentation
    // of existing code.

    // Uses std::shared_ptr<Foo3> as holder in Classic or Conservative mode, py::smart_holder in
    // Progressive mode.
    py::class_<Foo3, PYBIND11_SH_DEF(Foo3)>(m, "Foo3").def(py::init<>());
    // ------------- std::shared_ptr<Foo3> -- same length by design, to not disturb the indentation
    // of existing code.

    // Uses py::smart_holder if available, or std::unique_ptr<Foo3> if only pybind11 Classic is
    // available.
    py::classh<Foo4>(m, "Foo4").def(py::init<>());
}
