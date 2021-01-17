#include "pybind11_tests.h"

#include <pybind11/classh.h>

#include <memory>
#include <string>

namespace pybind11_tests {
namespace classh_wip {

struct mpty {
    std::string mtxt;
};

// clang-format off

mpty        rtrn_mpty_valu() { mpty obj{"rtrn_valu"}; return obj; }
mpty&&      rtrn_mpty_rref() { static mpty obj; obj.mtxt = "rtrn_rref"; return std::move(obj); }
mpty const& rtrn_mpty_cref() { static mpty obj; obj.mtxt = "rtrn_cref"; return obj; }
mpty&       rtrn_mpty_mref() { static mpty obj; obj.mtxt = "rtrn_mref"; return obj; }
mpty const* rtrn_mpty_cptr() { return new mpty{"rtrn_cptr"}; }
mpty*       rtrn_mpty_mptr() { return new mpty{"rtrn_mptr"}; }

std::string pass_mpty_valu(mpty obj)        { return "pass_valu:" + obj.mtxt; }
std::string pass_mpty_rref(mpty&& obj)      { return "pass_rref:" + obj.mtxt; }
std::string pass_mpty_cref(mpty const& obj) { return "pass_cref:" + obj.mtxt; }
std::string pass_mpty_mref(mpty& obj)       { return "pass_mref:" + obj.mtxt; }
std::string pass_mpty_cptr(mpty const* obj) { return "pass_cptr:" + obj->mtxt; }
std::string pass_mpty_mptr(mpty* obj)       { return "pass_mptr:" + obj->mtxt; }

std::shared_ptr<mpty>       rtrn_mpty_shmp() { return std::shared_ptr<mpty      >(new mpty{"rtrn_shmp"}); }
std::shared_ptr<mpty const> rtrn_mpty_shcp() { return std::shared_ptr<mpty const>(new mpty{"rtrn_shcp"}); }

std::string pass_mpty_shmp(std::shared_ptr<mpty>       obj) { return "pass_shmp:" + obj->mtxt; }
std::string pass_mpty_shcp(std::shared_ptr<mpty const> obj) { return "pass_shcp:" + obj->mtxt; }

std::unique_ptr<mpty>       rtrn_mpty_uqmp() { return std::unique_ptr<mpty      >(new mpty{"rtrn_uqmp"}); }
std::unique_ptr<mpty const> rtrn_mpty_uqcp() { return std::unique_ptr<mpty const>(new mpty{"rtrn_uqcp"}); }

std::string pass_mpty_uqmp(std::unique_ptr<mpty      > obj) { return "pass_uqmp:" + obj->mtxt; }
std::string pass_mpty_uqcp(std::unique_ptr<mpty const> obj) { return "pass_uqcp:" + obj->mtxt; }

// clang-format on

// Helpers for testing.
std::string get_mtxt(mpty const &obj) { return obj.mtxt; }
std::unique_ptr<mpty> unique_ptr_roundtrip(std::unique_ptr<mpty> obj) { return obj; }

} // namespace classh_wip
} // namespace pybind11_tests

PYBIND11_CLASSH_TYPE_CASTERS(pybind11_tests::classh_wip::mpty)

namespace pybind11_tests {
namespace classh_wip {

TEST_SUBMODULE(classh_wip, m) {
    namespace py = pybind11;

    py::classh<mpty>(m, "mpty").def(py::init<>()).def(py::init([](const std::string &mtxt) {
        mpty obj;
        obj.mtxt = mtxt;
        return obj;
    }));

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

    // Helpers for testing.
    // These require selected functions above to work first, as indicated:
    m.def("get_mtxt", get_mtxt);                         // pass_mpty_cref
    m.def("unique_ptr_roundtrip", unique_ptr_roundtrip); // pass_mpty_uqmp, rtrn_mpty_uqmp
}

} // namespace classh_wip
} // namespace pybind11_tests
