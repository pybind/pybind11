#include "pybind11_tests.h"

TEST_SUBMODULE(stubgen, m) {
    m.def("add_int", [](int a, int b) { return a + b; }, "a"_a, "b"_a);
}
