// Identical to classh_module_local_1.cpp, except 1 replaced with 2.
#include <pybind11/classh.h>

#include <string>

namespace pybind11_tests {
namespace classh_module_local {

struct bottle {
    std::string msg;
};

std::string get_msg(const bottle &b) { return b.msg; }

} // namespace classh_module_local
} // namespace pybind11_tests

PYBIND11_CLASSH_TYPE_CASTERS(pybind11_tests::classh_module_local::bottle)

PYBIND11_MODULE(classh_module_local_2, m) {
    namespace py = pybind11;
    using namespace pybind11_tests::classh_module_local;

    py::classh<bottle>(m, "bottle", py::module_local())
        .def(py::init([](const std::string &msg) {
            bottle obj;
            obj.msg = msg;
            return obj;
        }))
        .def("tag", [](const bottle &) { return 2; });

    m.def("get_msg", get_msg);
}
