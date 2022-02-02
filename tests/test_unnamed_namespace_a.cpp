#include "pybind11_tests.h"

namespace {
struct any_struct {};
} // namespace

TEST_SUBMODULE(unnamed_namespace_a, m) {
    py::class_<any_struct>(m, "unnamed_namespace_a_any_struct");
}
