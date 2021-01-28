#include "pybind11_tests.h"

#include <pybind11/smart_holder.h>

#include <memory>
#include <string>

namespace pybind11_tests {
namespace class_sh_basic {

struct atyp { // Short for "any type".
    std::string mtxt;
};

// clang-format off

atyp        rtrn_valu_atyp() { atyp obj{"rtrn_valu"}; return obj; }
atyp&&      rtrn_rref_atyp() { static atyp obj; obj.mtxt = "rtrn_rref"; return std::move(obj); }
atyp const& rtrn_cref_atyp() { static atyp obj; obj.mtxt = "rtrn_cref"; return obj; }
atyp&       rtrn_mref_atyp() { static atyp obj; obj.mtxt = "rtrn_mref"; return obj; }
atyp const* rtrn_cptr_atyp() { return new atyp{"rtrn_cptr"}; }
atyp*       rtrn_mptr_atyp() { return new atyp{"rtrn_mptr"}; }

std::string pass_valu_atyp(atyp obj)        { return "pass_valu:" + obj.mtxt; }
std::string pass_rref_atyp(atyp&& obj)      { return "pass_rref:" + obj.mtxt; }
std::string pass_cref_atyp(atyp const& obj) { return "pass_cref:" + obj.mtxt; }
std::string pass_mref_atyp(atyp& obj)       { return "pass_mref:" + obj.mtxt; }
std::string pass_cptr_atyp(atyp const* obj) { return "pass_cptr:" + obj->mtxt; }
std::string pass_mptr_atyp(atyp* obj)       { return "pass_mptr:" + obj->mtxt; }

std::shared_ptr<atyp>       rtrn_shmp_atyp() { return std::shared_ptr<atyp      >(new atyp{"rtrn_shmp"}); }
std::shared_ptr<atyp const> rtrn_shcp_atyp() { return std::shared_ptr<atyp const>(new atyp{"rtrn_shcp"}); }

std::string pass_shmp_atyp(std::shared_ptr<atyp>       obj) { return "pass_shmp:" + obj->mtxt; }
std::string pass_shcp_atyp(std::shared_ptr<atyp const> obj) { return "pass_shcp:" + obj->mtxt; }

std::unique_ptr<atyp>       rtrn_uqmp_atyp() { return std::unique_ptr<atyp      >(new atyp{"rtrn_uqmp"}); }
std::unique_ptr<atyp const> rtrn_uqcp_atyp() { return std::unique_ptr<atyp const>(new atyp{"rtrn_uqcp"}); }

std::string pass_uqmp_atyp(std::unique_ptr<atyp      > obj) { return "pass_uqmp:" + obj->mtxt; }
std::string pass_uqcp_atyp(std::unique_ptr<atyp const> obj) { return "pass_uqcp:" + obj->mtxt; }

// clang-format on

// Helpers for testing.
std::string get_mtxt(atyp const &obj) { return obj.mtxt; }
std::unique_ptr<atyp> unique_ptr_roundtrip(std::unique_ptr<atyp> obj) { return obj; }

} // namespace class_sh_basic
} // namespace pybind11_tests

PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_basic::atyp)

namespace pybind11_tests {
namespace class_sh_basic {

TEST_SUBMODULE(class_sh_basic, m) {
    namespace py = pybind11;

    py::classh<atyp>(m, "atyp").def(py::init<>()).def(py::init([](const std::string &mtxt) {
        atyp obj;
        obj.mtxt = mtxt;
        return obj;
    }));

    m.def("rtrn_valu_atyp", rtrn_valu_atyp);
    m.def("rtrn_rref_atyp", rtrn_rref_atyp);
    m.def("rtrn_cref_atyp", rtrn_cref_atyp);
    m.def("rtrn_mref_atyp", rtrn_mref_atyp);
    m.def("rtrn_cptr_atyp", rtrn_cptr_atyp);
    m.def("rtrn_mptr_atyp", rtrn_mptr_atyp);

    m.def("pass_valu_atyp", pass_valu_atyp);
    m.def("pass_rref_atyp", pass_rref_atyp);
    m.def("pass_cref_atyp", pass_cref_atyp);
    m.def("pass_mref_atyp", pass_mref_atyp);
    m.def("pass_cptr_atyp", pass_cptr_atyp);
    m.def("pass_mptr_atyp", pass_mptr_atyp);

    m.def("rtrn_shmp_atyp", rtrn_shmp_atyp);
    m.def("rtrn_shcp_atyp", rtrn_shcp_atyp);

    m.def("pass_shmp_atyp", pass_shmp_atyp);
    m.def("pass_shcp_atyp", pass_shcp_atyp);

    m.def("rtrn_uqmp_atyp", rtrn_uqmp_atyp);
    m.def("rtrn_uqcp_atyp", rtrn_uqcp_atyp);

    m.def("pass_uqmp_atyp", pass_uqmp_atyp);
    m.def("pass_uqcp_atyp", pass_uqcp_atyp);

    // Helpers for testing.
    // These require selected functions above to work first, as indicated:
    m.def("get_mtxt", get_mtxt);                         // pass_cref_atyp
    m.def("unique_ptr_roundtrip", unique_ptr_roundtrip); // pass_uqmp_atyp, rtrn_uqmp_atyp

    m.def("py_type_handle_of_atyp", []() {
        return py::type::handle_of<atyp>(); // Exercises static_cast in this function.
    });
}

} // namespace class_sh_basic
} // namespace pybind11_tests
