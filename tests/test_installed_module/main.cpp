#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_PLUGIN(test_installed_module) {
    py::module m("test_installed_module");

    m.def("add", [](int i, int j) { return i + j; });

    return m.ptr();
}
