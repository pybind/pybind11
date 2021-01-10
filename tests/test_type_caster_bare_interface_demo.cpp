#include "pybind11_tests.h"

namespace pybind11_tests {
namespace type_caster_bare_interface_demo {

struct mpty {};

mpty        rtrn_mpty_valu() { mpty obj; return obj; }
mpty&&      rtrn_mpty_rref() { mpty obj; return std::move(obj); }
mpty const& rtrn_mpty_cref() { static mpty obj; return obj; }
mpty&       rtrn_mpty_mref() { static mpty obj; return obj; }
mpty const* rtrn_mpty_cptr() { static mpty obj; return &obj; }
mpty*       rtrn_mpty_mptr() { static mpty obj; return &obj; }

const char* pass_mpty_valu(mpty)        { return "load_valu"; }
const char* pass_mpty_rref(mpty&&)      { return "load_rref"; }
const char* pass_mpty_cref(mpty const&) { return "load_cref"; }
const char* pass_mpty_mref(mpty&)       { return "load_mref"; }
const char* pass_mpty_cptr(mpty const*) { return "load_cptr"; }
const char* pass_mpty_mptr(mpty*)       { return "load_mptr"; }

}  // namespace type_caster_bare_interface_demo
}  // namespace pybind11_tests

namespace pybind11 {
namespace detail {

using namespace pybind11_tests::type_caster_bare_interface_demo;

template <>
struct type_caster<mpty> {
    static constexpr auto name = _<mpty>();

    // static handle cast(mpty, ...)
    // is redundant (leads to ambiguous overloads).

    static handle cast(mpty&& /*src*/,
                       return_value_policy /*policy*/,
                       handle /*parent*/) {
        return str("cast_rref").release();
    }

    static handle cast(mpty const& /*src*/,
                       return_value_policy /*policy*/,
                       handle /*parent*/) {
        return str("cast_cref").release();
    }

    static handle cast(mpty& /*src*/,
                       return_value_policy /*policy*/,
                       handle /*parent*/) {
        return str("cast_mref").release();
    }

    static handle cast(mpty const* /*src*/,
                       return_value_policy /*policy*/,
                       handle /*parent*/) {
        return str("cast_cptr").release();
    }

    static handle cast(mpty* /*src*/,
                       return_value_policy /*policy*/,
                       handle /*parent*/) {
        return str("cast_mptr").release();
    }

    template <typename T_>
    using cast_op_type = conditional_t<
        std::is_same<remove_reference_t<T_>, const mpty*>::value, const mpty*,
        conditional_t<
            std::is_same<remove_reference_t<T_>, mpty*>::value, mpty*,
            conditional_t<
                std::is_same<T_, const mpty&>::value, const mpty&,
                conditional_t<
                    std::is_same<T_, mpty&>::value, mpty&,
                    conditional_t<
                        std::is_same<T_, mpty&&>::value, mpty&&,
                        mpty>>>>>;

    operator mpty()        { return rtrn_mpty_valu(); }
    operator mpty&&() &&   { return rtrn_mpty_rref(); }
    operator mpty const&() { return rtrn_mpty_cref(); }
    operator mpty&()       { return rtrn_mpty_mref(); }
    operator mpty const*() { return rtrn_mpty_cptr(); }
    operator mpty*()       { return rtrn_mpty_mptr(); }

    bool load(handle /*src*/, bool /*convert*/) {
      return true;
    }
};

}  // namespace detail
}  // namespace pybind11

namespace pybind11_tests {
namespace type_caster_bare_interface_demo {

TEST_SUBMODULE(type_caster_bare_interface_demo, m) {
  m.def("rtrn_mpty_valu", rtrn_mpty_valu);
  m.def("rtrn_mpty_rref", rtrn_mpty_rref);
  m.def("rtrn_mpty_cref", rtrn_mpty_cref);
  m.def("rtrn_mpty_mref", rtrn_mpty_mref);
  m.def("rtrn_mpty_cptr", rtrn_mpty_cptr);
  m.def("rtrn_mpty_mptr", rtrn_mpty_mptr);

  m.def("pass_mpty_valu", pass_mpty_valu);
  m.def("pass_mpty_rref", pass_mpty_rref);
  m.def("pass_mpty_cref", pass_mpty_cref);
  m.def("pass_mpty_mref", pass_mpty_mref);
  m.def("pass_mpty_cptr", pass_mpty_cptr);
  m.def("pass_mpty_mptr", pass_mpty_mptr);
}

}  // namespace type_caster_bare_interface_demo
}  // namespace pybind11_tests
