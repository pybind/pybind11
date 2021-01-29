// Systematically exercises the detail::type_caster<> interface. This is going a step in the
// direction of an integration test, to ensure multiple components of pybind11 work together
// correctly. It is also useful to show the type_caster<> interface virtually clutter-free.

#include "pybind11_tests.h"

#include <memory>

namespace pybind11_tests {
namespace type_caster_bare_interface {

struct mpty {};

// clang-format off

mpty        rtrn_valu() { mpty obj; return obj; }
mpty&&      rtrn_rref() { static mpty obj; return std::move(obj); }
mpty const& rtrn_cref() { static mpty obj; return obj; }
mpty&       rtrn_mref() { static mpty obj; return obj; }
mpty const* rtrn_cptr() { return new mpty; }
mpty*       rtrn_mptr() { return new mpty; }

const char* pass_valu(mpty)        { return "load_valu"; }
const char* pass_rref(mpty&&)      { return "load_rref"; }
const char* pass_cref(mpty const&) { return "load_cref"; }
const char* pass_mref(mpty&)       { return "load_mref"; }
const char* pass_cptr(mpty const*) { return "load_cptr"; }
const char* pass_mptr(mpty*)       { return "load_mptr"; }

std::shared_ptr<mpty>       rtrn_shmp() { return std::shared_ptr<mpty      >(new mpty); }
std::shared_ptr<mpty const> rtrn_shcp() { return std::shared_ptr<mpty const>(new mpty); }

const char* pass_shmp(std::shared_ptr<mpty>)       { return "load_shmp"; }
const char* pass_shcp(std::shared_ptr<mpty const>) { return "load_shcp"; }

std::unique_ptr<mpty>       rtrn_uqmp() { return std::unique_ptr<mpty      >(new mpty); }
std::unique_ptr<mpty const> rtrn_uqcp() { return std::unique_ptr<mpty const>(new mpty); }

const char* pass_uqmp(std::unique_ptr<mpty>)       { return "load_uqmp"; }
const char* pass_uqcp(std::unique_ptr<mpty const>) { return "load_uqcp"; }

// clang-format on

} // namespace type_caster_bare_interface
} // namespace pybind11_tests

namespace pybind11 {
namespace detail {

using namespace pybind11_tests::type_caster_bare_interface;

template <>
struct type_caster<mpty> {
    static constexpr auto name = _<mpty>();

    // static handle cast(mpty, ...)
    // is redundant (leads to ambiguous overloads).

    static handle cast(mpty && /*src*/, return_value_policy /*policy*/, handle /*parent*/) {
        return str("cast_rref").release();
    }

    static handle cast(mpty const & /*src*/, return_value_policy /*policy*/, handle /*parent*/) {
        return str("cast_cref").release();
    }

    static handle cast(mpty & /*src*/, return_value_policy /*policy*/, handle /*parent*/) {
        return str("cast_mref").release();
    }

    static handle cast(mpty const *src, return_value_policy /*policy*/, handle /*parent*/) {
        delete src;
        return str("cast_cptr").release();
    }

    static handle cast(mpty *src, return_value_policy /*policy*/, handle /*parent*/) {
        delete src;
        return str("cast_mptr").release();
    }

    template <typename T_>
    using cast_op_type = conditional_t<
        std::is_same<remove_reference_t<T_>, mpty const *>::value,
        mpty const *,
        conditional_t<
            std::is_same<remove_reference_t<T_>, mpty *>::value,
            mpty *,
            conditional_t<
                std::is_same<T_, mpty const &>::value,
                mpty const &,
                conditional_t<std::is_same<T_, mpty &>::value,
                              mpty &,
                              conditional_t<std::is_same<T_, mpty &&>::value, mpty &&, mpty>>>>>;

    // clang-format off

    operator mpty()        { return rtrn_valu(); }
    operator mpty&&() &&   { return rtrn_rref(); }
    operator mpty const&() { return rtrn_cref(); }
    operator mpty&()       { return rtrn_mref(); }
    operator mpty const*() { static mpty obj; return &obj; }
    operator mpty*()       { static mpty obj; return &obj; }

    // clang-format on

    bool load(handle /*src*/, bool /*convert*/) { return true; }
};

template <>
struct type_caster<std::shared_ptr<mpty>> {
    static constexpr auto name = _<std::shared_ptr<mpty>>();

    static handle cast(const std::shared_ptr<mpty> & /*src*/,
                       return_value_policy /*policy*/,
                       handle /*parent*/) {
        return str("cast_shmp").release();
    }

    template <typename>
    using cast_op_type = std::shared_ptr<mpty>;

    operator std::shared_ptr<mpty>() { return rtrn_shmp(); }

    bool load(handle /*src*/, bool /*convert*/) { return true; }
};

template <>
struct type_caster<std::shared_ptr<mpty const>> {
    static constexpr auto name = _<std::shared_ptr<mpty const>>();

    static handle cast(const std::shared_ptr<mpty const> & /*src*/,
                       return_value_policy /*policy*/,
                       handle /*parent*/) {
        return str("cast_shcp").release();
    }

    template <typename>
    using cast_op_type = std::shared_ptr<mpty const>;

    operator std::shared_ptr<mpty const>() { return rtrn_shcp(); }

    bool load(handle /*src*/, bool /*convert*/) { return true; }
};

template <>
struct type_caster<std::unique_ptr<mpty>> {
    static constexpr auto name = _<std::unique_ptr<mpty>>();

    static handle
    cast(std::unique_ptr<mpty> && /*src*/, return_value_policy /*policy*/, handle /*parent*/) {
        return str("cast_uqmp").release();
    }

    template <typename>
    using cast_op_type = std::unique_ptr<mpty>;

    operator std::unique_ptr<mpty>() { return rtrn_uqmp(); }

    bool load(handle /*src*/, bool /*convert*/) { return true; }
};

template <>
struct type_caster<std::unique_ptr<mpty const>> {
    static constexpr auto name = _<std::unique_ptr<mpty const>>();

    static handle cast(std::unique_ptr<mpty const> && /*src*/,
                       return_value_policy /*policy*/,
                       handle /*parent*/) {
        return str("cast_uqcp").release();
    }

    template <typename>
    using cast_op_type = std::unique_ptr<mpty const>;

    operator std::unique_ptr<mpty const>() { return rtrn_uqcp(); }

    bool load(handle /*src*/, bool /*convert*/) { return true; }
};

} // namespace detail
} // namespace pybind11

namespace pybind11_tests {
namespace type_caster_bare_interface {

TEST_SUBMODULE(type_caster_bare_interface, m) {
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

    m.def("rtrn_shmp", rtrn_shmp);
    m.def("rtrn_shcp", rtrn_shcp);

    m.def("pass_shmp", pass_shmp);
    m.def("pass_shcp", pass_shcp);

    m.def("rtrn_uqmp", rtrn_uqmp);
    m.def("rtrn_uqcp", rtrn_uqcp);

    m.def("pass_uqmp", pass_uqmp);
    m.def("pass_uqcp", pass_uqcp);
}

} // namespace type_caster_bare_interface
} // namespace pybind11_tests
