#include <pybind11/classh.h>

#include <string>

namespace pybind11_tests {
namespace classh_module_local {

struct bottle {
    std::string msg;
};

std::string get_msg(const bottle &b) { return b.msg; }

bottle make_bottle() { return bottle(); }

} // namespace classh_module_local
} // namespace pybind11_tests

PYBIND11_CLASSH_TYPE_CASTERS(pybind11_tests::classh_module_local::bottle)

PYBIND11_MODULE(classh_module_local_0, m) {
    using namespace pybind11_tests::classh_module_local;

    m.def("get_msg", get_msg);

    m.def("make_bottle", make_bottle);
}
