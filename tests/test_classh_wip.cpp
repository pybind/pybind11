#include "pybind11_tests.h"

#include <pybind11/classh.h>

#include <memory>
#include <string>

namespace pybind11_tests {
namespace classh_wip {

struct mpty { std::string mtxt; };

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

std::shared_ptr<mpty>       rtrn_mpty_shmp() { return std::shared_ptr<mpty>(new mpty); }
std::shared_ptr<mpty const> rtrn_mpty_shcp() { return std::shared_ptr<mpty const>(new mpty); }

const char* pass_mpty_shmp(std::shared_ptr<mpty>)       { return "load_shmp"; }
const char* pass_mpty_shcp(std::shared_ptr<mpty const>) { return "load_shcp"; }

std::unique_ptr<mpty>       rtrn_mpty_uqmp() { return std::unique_ptr<mpty>(new mpty); }
std::unique_ptr<mpty const> rtrn_mpty_uqcp() { return std::unique_ptr<mpty const>(new mpty); }

const char* pass_mpty_uqmp(std::unique_ptr<mpty>)       { return "load_uqmp"; }
const char* pass_mpty_uqcp(std::unique_ptr<mpty const>) { return "load_uqcp"; }

}  // namespace classh_wip
}  // namespace pybind11_tests

namespace pybind11 {
namespace detail {

using namespace pybind11_tests::classh_wip;

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
        std::is_same<remove_reference_t<T_>, mpty const*>::value, mpty const*,
        conditional_t<
            std::is_same<remove_reference_t<T_>, mpty*>::value, mpty*,
            conditional_t<
                std::is_same<T_, mpty const&>::value, mpty const&,
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

template <>
struct type_caster<std::shared_ptr<mpty>> {
    static constexpr auto name = _<std::shared_ptr<mpty>>();

    static handle cast(const std::shared_ptr<mpty>& /*src*/,
                       return_value_policy /*policy*/,
                       handle /*parent*/) {
        return str("cast_shmp").release();
    }

    template <typename> using cast_op_type = std::shared_ptr<mpty>;

    operator std::shared_ptr<mpty>() { return rtrn_mpty_shmp(); }

    bool load(handle /*src*/, bool /*convert*/) {
        return true;
    }
};

template <>
struct type_caster<std::shared_ptr<mpty const>> {
    static constexpr auto name = _<std::shared_ptr<mpty const>>();

    static handle cast(const std::shared_ptr<mpty const>& /*src*/,
                       return_value_policy /*policy*/,
                       handle /*parent*/) {
        return str("cast_shcp").release();
    }

    template <typename> using cast_op_type = std::shared_ptr<mpty const>;

    operator std::shared_ptr<mpty const>() { return rtrn_mpty_shcp(); }

    bool load(handle /*src*/, bool /*convert*/) {
        return true;
    }
};

template <>
struct type_caster<std::unique_ptr<mpty>> {
    static constexpr auto name = _<std::unique_ptr<mpty>>();

    static handle cast(std::unique_ptr<mpty>&& /*src*/,
                       return_value_policy /*policy*/,
                       handle /*parent*/) {
        return str("cast_uqmp").release();
    }

    template <typename> using cast_op_type = std::unique_ptr<mpty>;

    operator std::unique_ptr<mpty>() { return rtrn_mpty_uqmp(); }

    bool load(handle /*src*/, bool /*convert*/) {
        return true;
    }
};

template <>
struct type_caster<std::unique_ptr<mpty const>> {
    static constexpr auto name = _<std::unique_ptr<mpty const>>();

    static handle cast(std::unique_ptr<mpty const>&& /*src*/,
                       return_value_policy /*policy*/,
                       handle /*parent*/) {
        return str("cast_uqcp").release();
    }

    template <typename> using cast_op_type = std::unique_ptr<mpty const>;

    operator std::unique_ptr<mpty const>() { return rtrn_mpty_uqcp(); }

    bool load(handle /*src*/, bool /*convert*/) {
        return true;
    }
};

}  // namespace detail
}  // namespace pybind11

namespace pybind11_tests {
namespace classh_wip {

TEST_SUBMODULE(classh_wip, m) {
    namespace py = pybind11;

    py::classh<mpty>(m, "mpty")
        .def(py::init<>())
        .def(py::init([](const std::string& mtxt) {
            mpty obj; obj.mtxt = mtxt; return obj; }))
    ;

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

    m.def("rtrn_mpty_shmp", rtrn_mpty_shmp);
    m.def("rtrn_mpty_shcp", rtrn_mpty_shcp);

    m.def("pass_mpty_shmp", pass_mpty_shmp);
    m.def("pass_mpty_shcp", pass_mpty_shcp);

    m.def("rtrn_mpty_uqmp", rtrn_mpty_uqmp);
    m.def("rtrn_mpty_uqcp", rtrn_mpty_uqcp);

    m.def("pass_mpty_uqmp", pass_mpty_uqmp);
    m.def("pass_mpty_uqcp", pass_mpty_uqcp);
}

}  // namespace classh_wip
}  // namespace pybind11_tests
