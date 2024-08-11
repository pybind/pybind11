#if defined(PYBIND11_INTERNALS_VERSION)
#    undef PYBIND11_INTERNALS_VERSION
#endif
#define PYBIND11_INTERNALS_VERSION 900000001

#include "test_cpp_transporter_traveler_bindings.h"

PYBIND11_MODULE(exo_planet, m) { pybind11_tests::test_cpp_transporter::wrap_traveler(m); }
