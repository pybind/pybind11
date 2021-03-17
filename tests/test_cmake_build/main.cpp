// clang-format off
// Please consider `clang-format`ting this file before starting with significant
// changes, ideally in a separate commit. See also: .github/CONTRIBUTING.md

#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(test_cmake_build, m) {
    m.def("add", [](int i, int j) { return i + j; });
}
