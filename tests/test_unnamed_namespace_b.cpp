#include "pybind11_tests.h"

namespace {
struct any_struct {};
} // namespace

TEST_SUBMODULE(unnamed_namespace_b, m) {
    py::class_<any_struct>(m, "unnamed_namespace_b_any_struct");
}
