#include "pybind11_tests.h"

class SimpleClass {};

TEST_SUBMODULE(keep_alive, m) {
    m.def("keep_alive_impl", [](py::handle nurse, py::handle patient) {
        py::detail::keep_alive_impl(nurse, patient);
    });

    py::class_<SimpleClass>(m, "SimpleClass")
        .def(py::init());
}
