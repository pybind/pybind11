#include "pybind11_tests.h"

#include <pybind11/smart_holder.h>

#include <memory>
#include <string>

namespace pybind11_tests {
namespace class_sh_basic {

struct atyp { // Short for "any type".
    std::string mtxt;
    atyp() : mtxt("DefaultConstructor") {}
    atyp(const std::string &mtxt_) : mtxt(mtxt_) {}
    atyp(const atyp &other) { mtxt = other.mtxt + "_CpCtor"; }
    atyp(atyp &&other) { mtxt = other.mtxt + "_MvCtor"; }
};

// clang-format off

atyp        rtrn_valu() { atyp obj{"rtrn_valu"}; return obj; }
atyp&&      rtrn_rref() { static atyp obj; obj.mtxt = "rtrn_rref"; return std::move(obj); }
atyp const& rtrn_cref() { static atyp obj; obj.mtxt = "rtrn_cref"; return obj; }
atyp&       rtrn_mref() { static atyp obj; obj.mtxt = "rtrn_mref"; return obj; }
atyp const* rtrn_cptr() { return new atyp{"rtrn_cptr"}; }
atyp*       rtrn_mptr() { return new atyp{"rtrn_mptr"}; }

std::string pass_valu(atyp obj)        { return "pass_valu:" + obj.mtxt; }
std::string pass_rref(atyp&& obj)      { return "pass_rref:" + obj.mtxt; }
std::string pass_cref(atyp const& obj) { return "pass_cref:" + obj.mtxt; }
std::string pass_mref(atyp& obj)       { return "pass_mref:" + obj.mtxt; }
std::string pass_cptr(atyp const* obj) { return "pass_cptr:" + obj->mtxt; }
std::string pass_mptr(atyp* obj)       { return "pass_mptr:" + obj->mtxt; }

std::shared_ptr<atyp>       rtrn_shmp() { return std::shared_ptr<atyp      >(new atyp{"rtrn_shmp"}); }
std::shared_ptr<atyp const> rtrn_shcp() { return std::shared_ptr<atyp const>(new atyp{"rtrn_shcp"}); }

std::string pass_shmp(std::shared_ptr<atyp>       obj) { return "pass_shmp:" + obj->mtxt; }
std::string pass_shcp(std::shared_ptr<atyp const> obj) { return "pass_shcp:" + obj->mtxt; }

std::unique_ptr<atyp>       rtrn_uqmp() { return std::unique_ptr<atyp      >(new atyp{"rtrn_uqmp"}); }
std::unique_ptr<atyp const> rtrn_uqcp() { return std::unique_ptr<atyp const>(new atyp{"rtrn_uqcp"}); }

std::string pass_uqmp(std::unique_ptr<atyp      > obj) { return "pass_uqmp:" + obj->mtxt; }
std::string pass_uqcp(std::unique_ptr<atyp const> obj) { return "pass_uqcp:" + obj->mtxt; }

struct sddm : std::default_delete<atyp      > {};
struct sddc : std::default_delete<atyp const> {};

std::unique_ptr<atyp,       sddm> rtrn_udmp() { return std::unique_ptr<atyp,       sddm>(new atyp{"rtrn_udmp"}); }
std::unique_ptr<atyp const, sddc> rtrn_udcp() { return std::unique_ptr<atyp const, sddc>(new atyp{"rtrn_udcp"}); }

std::string pass_udmp(std::unique_ptr<atyp,       sddm> obj) { return "pass_udmp:" + obj->mtxt; }
std::string pass_udcp(std::unique_ptr<atyp const, sddc> obj) { return "pass_udcp:" + obj->mtxt; }

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

    m.def("rtrn_udmp", rtrn_udmp);
    m.def("rtrn_udcp", rtrn_udcp);

    m.def("pass_udmp", pass_udmp);
    m.def("pass_udcp", pass_udcp);

    // Helpers for testing.
    // These require selected functions above to work first, as indicated:
    m.def("get_mtxt", get_mtxt);                         // pass_cref
    m.def("unique_ptr_roundtrip", unique_ptr_roundtrip); // pass_uqmp, rtrn_uqmp

    m.def("py_type_handle_of_atyp", []() {
        return py::type::handle_of<atyp>(); // Exercises static_cast in this function.
    });
}

} // namespace class_sh_basic
} // namespace pybind11_tests
