#include "pybind11/pybind11.h"

PYBIND11_MODULE(cross_module_exception_odr_1, m) { m.attr("foo") = 1; }
