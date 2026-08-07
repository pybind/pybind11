// Copyright (c) 2025 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

// Single-TU build of the pybind11 library sources, for build systems that prefer adding
// one file over one file per header (e.g. setuptools). Compile this file (and every TU
// that includes pybind11) with PYBIND11_PRECOMPILED defined. One include per sibling
// src/*.cpp file; the CMake path compiles those files individually instead.

#if !defined(PYBIND11_PRECOMPILED)
#    error "pybind11 library sources must be compiled with PYBIND11_PRECOMPILED defined."
#endif

#include "class.cpp"
#include "common.cpp"
#include "exception_translation.cpp"
#include "internals.cpp"
#include "pybind11.cpp"
#include "pytypes.cpp"
#include "type_caster_base.cpp"
