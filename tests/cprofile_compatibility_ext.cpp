#include <pybind11/pybind11.h>

namespace pybind11_tests {
namespace cprofile_compatibility {
class CppClass {};
} // namespace cprofile_compatibility
} // namespace pybind11_tests

namespace py = pybind11;

PYBIND11_MODULE(cprofile_compatibility_ext, m) {
    using namespace pybind11_tests::cprofile_compatibility;

    m.def("free_func_return_secret", []() { return 102; });

    py::class_<CppClass>(m, "CppClass")
        .def(py::init<>())
        .def("member_func_return_secret", [](const CppClass &) { return 203; });
}
