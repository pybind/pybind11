// Identical to class_sh_module_local_1.cpp, except 1 replaced with 2.
#include <pybind11/smart_holder.h>

#include <string>

namespace pybind11_tests {
namespace class_sh_module_local {

struct atyp { // Short for "any type".
    std::string mtxt;
};

std::string get_mtxt(const atyp &obj) { return obj.mtxt; }

} // namespace class_sh_module_local
} // namespace pybind11_tests

PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_module_local::atyp)

PYBIND11_MODULE(class_sh_module_local_2, m) {
    namespace py = pybind11;
    using namespace pybind11_tests::class_sh_module_local;

    py::classh<atyp>(m, "atyp", py::module_local())
        .def(py::init([](const std::string &mtxt) {
            atyp obj;
            obj.mtxt = mtxt;
            return obj;
        }))
        .def("tag", [](const atyp &) { return 2; });

    m.def("get_mtxt", get_mtxt);
}
