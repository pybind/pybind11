#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(test_cmake_build, m) {
    m.set_gil_not_used();
    m.def("add", [](int i, int j) { return i + j; });
}
