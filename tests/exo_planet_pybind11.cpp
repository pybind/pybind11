// Copyright (c) 2024 The pybind Community.

#if defined(PYBIND11_INTERNALS_VERSION)
#    undef PYBIND11_INTERNALS_VERSION
#endif
#define PYBIND11_INTERNALS_VERSION 900000001

#include "test_cpp_conduit_traveler_bindings.h"

PYBIND11_MODULE(exo_planet_pybind11, m) { pybind11_tests::test_cpp_conduit::wrap_traveler(m); }
