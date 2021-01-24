#include <pybind11/classh.h>
#include <pybind11/pybind11.h>

#include <string>

namespace pybind11_tests {
namespace classh_module_local {

struct atyp { // Short for "any type".
    std::string mtxt;
};

std::string get_mtxt(const atyp &obj) { return obj.mtxt; }

atyp rtrn_valu_atyp() { return atyp(); }

} // namespace classh_module_local
} // namespace pybind11_tests

PYBIND11_CLASSH_TYPE_CASTERS(pybind11_tests::classh_module_local::atyp)

PYBIND11_MODULE(classh_module_local_0, m) {
    using namespace pybind11_tests::classh_module_local;

    m.def("get_mtxt", get_mtxt);

    m.def("rtrn_valu_atyp", rtrn_valu_atyp);
}
