// Copyright (c) 2024 The pybind Community.

#pragma once

#include "common.h"

/// Tracks the `internals` and `type_info` ABI version independent of the main library version.
///
/// Some portions of the code use an ABI that is conditional depending on this
/// version number.  That allows ABI-breaking changes to be "pre-implemented".
/// Once the default version number is incremented, the conditional logic that
/// no longer applies can be removed.  Additionally, users that need not
/// maintain ABI compatibility can increase the version number in order to take
/// advantage of any functionality/efficiency improvements that depend on the
/// newer ABI.
///
/// WARNING: If you choose to manually increase the ABI version, note that
/// pybind11 may not be tested as thoroughly with a non-default ABI version, and
/// further ABI-incompatible changes may be made before the ABI is officially
/// changed to the new version.
#ifndef PYBIND11_INTERNALS_VERSION
#    if PY_VERSION_HEX >= 0x030C0000 || defined(_MSC_VER)
// Version bump for Python 3.12+, before first 3.12 beta release.
// Version bump for MSVC piggy-backed on PR #4779. See comments there.
#        define PYBIND11_INTERNALS_VERSION 5
#    else
#        define PYBIND11_INTERNALS_VERSION 4
#    endif
#endif

// This requirement is mainly to reduce the support burden (see PR #4570).
static_assert(PY_VERSION_HEX < 0x030C0000 || PYBIND11_INTERNALS_VERSION >= 5,
              "pybind11 ABI version 5 is the minimum for Python 3.12+");
