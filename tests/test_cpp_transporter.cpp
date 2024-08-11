#include "pybind11_tests.h"
#include "test_cpp_transporter_traveler_bindings.h"

TEST_SUBMODULE(cpp_transporter, m) { pybind11_tests::test_cpp_transporter::wrap_traveler(m); }
