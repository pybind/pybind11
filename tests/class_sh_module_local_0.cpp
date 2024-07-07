#include <pybind11/smart_holder.h>

#include <string>

namespace pybind11_tests {
namespace class_sh_module_local {

struct atyp { // Short for "any type".
    std::string mtxt;
};

std::string get_mtxt(const atyp &obj) { return obj.mtxt; }

atyp rtrn_valu_atyp() { return atyp(); }

} // namespace class_sh_module_local
} // namespace pybind11_tests

PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_module_local::atyp)

PYBIND11_MODULE(class_sh_module_local_0, m) {
    using namespace pybind11_tests::class_sh_module_local;

    m.def("get_mtxt", get_mtxt);

    m.def("rtrn_valu_atyp", rtrn_valu_atyp);
}
