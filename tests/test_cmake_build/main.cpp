#include <pybind11/pybind11.h>

#ifdef EXPECT_PYBIND11_DISABLE_HANDLE_TYPE_NAME_DEFAULT_IMPLEMENTATION
#    ifndef PYBIND11_DISABLE_HANDLE_TYPE_NAME_DEFAULT_IMPLEMENTATION
#        error "PYBIND11_DISABLE_HANDLE_TYPE_NAME_DEFAULT_IMPLEMENTATION did not propagate"
#    endif
#endif
#ifdef EXPECT_PYBIND11_SIMPLE_GIL_MANAGEMENT
#    ifndef PYBIND11_SIMPLE_GIL_MANAGEMENT
#        error "PYBIND11_SIMPLE_GIL_MANAGEMENT did not propagate"
#    endif
#endif

namespace py = pybind11;

PYBIND11_MODULE(test_cmake_build, m, py::mod_gil_not_used()) {
    m.def("add", [](int i, int j) { return i + j; });
}
