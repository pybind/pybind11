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

struct ColdMethodType {
    int value() const { return 7; }
};

PYBIND11_MODULE(test_cmake_build, m, py::mod_gil_not_used()) {
    m.def("add", [](int i, int j) { return i + j; });
    py::class_<ColdMethodType>(m, "ColdMethodType")
        .def(py::init<>())
        .def("value", &ColdMethodType::value);
#if defined(Py_LIMITED_API)
    m.attr("limited_api") = true;
#else
    m.attr("limited_api") = false;
#endif
}
