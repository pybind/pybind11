// Systematically exercises the detail::type_caster<> interface. This is going a step in the
// direction of an integration test, to ensure multiple components of pybind11 work together
// correctly. It is also useful to show the type_caster<> interface virtually clutter-free.

// The entire type_caster load logic is intentionally omitted. The only purpose of this test is to
// trace the behavior of the `static handle cast()` functions and the type_caster `operator`s.
// Variable names are intentionally terse, to not distract from the more important function
// signatures: valu(e), ref(erence), ptr (pointer), r = rvalue, m = mutable, c = const.

#include "pybind11_tests.h"

#include <string>

namespace pybind11_tests {
namespace type_caster_bare_interface {

struct atyp { // Short for "any type".
    std::string trace;
    atyp() : trace("default") {}
    atyp(atyp const &other) { trace = other.trace + "_CpCtor"; }
    atyp(atyp &&other) { trace = other.trace + "_MvCtor"; }
};

// clang-format off

atyp        rtrn_valu() { static atyp obj; obj.trace = "valu"; return obj; }
atyp&&      rtrn_rref() { static atyp obj; obj.trace = "rref"; return std::move(obj); }
atyp const& rtrn_cref() { static atyp obj; obj.trace = "cref"; return obj; }
atyp&       rtrn_mref() { static atyp obj; obj.trace = "mref"; return obj; }
atyp const* rtrn_cptr() { static atyp obj; obj.trace = "cptr"; return &obj; }
atyp*       rtrn_mptr() { static atyp obj; obj.trace = "mptr"; return &obj; }

std::string pass_valu(atyp obj)        { return "pass_valu:" + obj.trace; }
std::string pass_rref(atyp&& obj)      { return "pass_rref:" + obj.trace; }
std::string pass_cref(atyp const& obj) { return "pass_cref:" + obj.trace; }
std::string pass_mref(atyp& obj)       { return "pass_mref:" + obj.trace; }
std::string pass_cptr(atyp const* obj) { return "pass_cptr:" + obj->trace; }
std::string pass_mptr(atyp* obj)       { return "pass_mptr:" + obj->trace; }

// clang-format on

} // namespace type_caster_bare_interface
} // namespace pybind11_tests

namespace pybind11 {
namespace detail {

using namespace pybind11_tests::type_caster_bare_interface;

template <>
struct type_caster<atyp> {
    static constexpr auto name = _<atyp>();

    // static handle cast(atyp, ...)
    // is redundant (leads to ambiguous overloads).

    static handle cast(atyp &&src, return_value_policy /*policy*/, handle /*parent*/) {
        return str("cast_rref:" + src.trace).release();
    }

    static handle cast(atyp const &src, return_value_policy /*policy*/, handle /*parent*/) {
        return str("cast_cref:" + src.trace).release();
    }

    static handle cast(atyp &src, return_value_policy /*policy*/, handle /*parent*/) {
        return str("cast_mref:" + src.trace).release();
    }

    static handle cast(atyp const *src, return_value_policy /*policy*/, handle /*parent*/) {
        return str("cast_cptr:" + src->trace).release();
    }

    static handle cast(atyp *src, return_value_policy /*policy*/, handle /*parent*/) {
        return str("cast_mptr:" + src->trace).release();
    }

    template <typename T_>
    using cast_op_type = conditional_t<
        std::is_same<remove_reference_t<T_>, atyp const *>::value,
        atyp const *,
        conditional_t<
            std::is_same<remove_reference_t<T_>, atyp *>::value,
            atyp *,
            conditional_t<
                std::is_same<T_, atyp const &>::value,
                atyp const &,
                conditional_t<std::is_same<T_, atyp &>::value,
                              atyp &,
                              conditional_t<std::is_same<T_, atyp &&>::value, atyp &&, atyp>>>>>;

    // clang-format off

    operator atyp()        { static atyp obj; obj.trace = "valu"; return obj; }
    operator atyp&&()      { static atyp obj; obj.trace = "rref"; return std::move(obj); }
    operator atyp const&() { static atyp obj; obj.trace = "cref"; return obj; }
    operator atyp&()       { static atyp obj; obj.trace = "mref"; return obj; }
    operator atyp const*() { static atyp obj; obj.trace = "cptr"; return &obj; }
    operator atyp*()       { static atyp obj; obj.trace = "mptr"; return &obj; }

    // clang-format on

    // The entire load logic is intentionally omitted.
    bool load(handle /*src*/, bool /*convert*/) { return true; }
};

} // namespace detail
} // namespace pybind11

TEST_SUBMODULE(type_caster_bare_interface, m) {
    namespace py = pybind11;
    using namespace pybind11_tests::type_caster_bare_interface;

    m.def("rtrn_valu", rtrn_valu);
    m.def("rtrn_rref", rtrn_rref);
    m.def("rtrn_cref", rtrn_cref);
    m.def("rtrn_mref", rtrn_mref);
    m.def("rtrn_cptr", rtrn_cptr);
    m.def("rtrn_mptr", rtrn_mptr);

    m.def("pass_valu", pass_valu);
    m.def("pass_rref", pass_rref);
    m.def("pass_cref", pass_cref);
    m.def("pass_mref", pass_mref);
    m.def("pass_cptr", pass_cptr);
    m.def("pass_mptr", pass_mptr);
}
