// Identical to classh_module_local_1.cpp, except 1 replaced with 2.
#include <pybind11/pybind11.h>
#include <pybind11/smart_holder.h>

#include <string>

namespace pybind11_tests {
namespace classh_module_local {

struct atyp { // Short for "any type".
    std::string mtxt;
};

std::string get_mtxt(const atyp &obj) { return obj.mtxt; }

} // namespace classh_module_local
} // namespace pybind11_tests

PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::classh_module_local::atyp)

PYBIND11_MODULE(classh_module_local_2, m) {
    namespace py = pybind11;
    using namespace pybind11_tests::classh_module_local;

    py::class_<atyp, py::smart_holder>(m, "atyp", py::module_local())
        .def(py::init([](const std::string &mtxt) {
            atyp obj;
            obj.mtxt = mtxt;
            return obj;
        }))
        .def("tag", [](const atyp &) { return 2; });

    m.def("get_mtxt", get_mtxt);
}
