#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_PLUGIN(test_cmake_build) {
    py::module m("test_cmake_build");

    m.def("add", [](int i, int j) { return i + j; });

    return m.ptr();
}
