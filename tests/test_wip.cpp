#include "pybind11_tests.h"

TEST_SUBMODULE(wip, m) { m.attr("__doc__") = "WIP"; }
