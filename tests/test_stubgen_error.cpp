#include "pybind11_tests.h"

TEST_SUBMODULE(stubgen_error, m) {
    m.def("identity_capsule", [](py::capsule c) { return c; }, "c"_a);
}
