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
struct FooUc {};
struct FooUp {};
struct FooSa {};
struct FooSc {};
struct FooSp {};
} // namespace

PYBIND11_SMART_HOLDER_TYPE_CASTERS(FooUp)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(FooSp)

PYBIND11_TYPE_CASTER_BASE_HOLDER(FooSa, std::shared_ptr<FooSa>)

TEST_SUBMODULE(classh_mock, m) {
    // Please see README_smart_holder.rst, in particular section
    // Classic / Conservative / Progressive cross-module compatibility

    // Uses std::unique_ptr<FooUc> as holder in Classic or Conservative mode, py::smart_holder in
    // Progressive mode.
    py::class_<FooUc>(m, "FooUc").def(py::init<>());

    // Uses std::unique_ptr<FooUp> as holder in Classic mode, py::smart_holder in Conservative or
    // Progressive mode.
    py::classh<FooUp>(m, "FooUp").def(py::init<>());

    // Always uses std::shared_ptr<FooSa> as holder.
    py::class_<FooSa, std::shared_ptr<FooSa>>(m, "FooSa").def(py::init<>());

    // Uses std::shared_ptr<FooSc> as holder in Classic or Conservative mode, py::smart_holder in
    // Progressive mode.
    py::class_<FooSc, PYBIND11_SH_DEF(FooSc)>(m, "FooSc").def(py::init<>());
    // -------------- std::shared_ptr<FooSc> -- same length by design, to not disturb the
    // indentation of existing code.

    // Uses std::shared_ptr<FooSp> as holder in Classic mode, py::smart_holder in Conservative or
    // Progressive mode.
    py::class_<FooSp, PYBIND11_SH_AVL(FooSp)>(m, "FooSp").def(py::init<>());
    // -------------- std::shared_ptr<FooSp> -- same length by design, to not disturb the
    // indentation of existing code.
}
