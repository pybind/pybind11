// Copyright (c) 2025 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

// Single-TU build of the pybind11 library sources, for build systems that prefer adding
// one file over one file per header (e.g. setuptools). Compile this file (and every TU
// that includes pybind11) with PYBIND11_PRECOMPILED defined. Keep in sync with the list
// of -inl.h files; the CMake path compiles the individual src/*.cpp files instead.

#if !defined(PYBIND11_PRECOMPILED)
#    error "pybind11 library sources must be compiled with PYBIND11_PRECOMPILED defined."
#endif

#include <pybind11/detail/internals-inl.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes-inl.h>
