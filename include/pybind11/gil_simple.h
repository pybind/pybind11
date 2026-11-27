// Copyright (c) 2016-2025 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include "detail/common.h"

#include <cassert>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

class gil_scoped_acquire_simple {
    PyGILState_STATE state;
    bool active = true;

public:
    gil_scoped_acquire_simple() : state{PyGILState_Ensure()} {}
    gil_scoped_acquire_simple(const gil_scoped_acquire_simple &) = delete;
    gil_scoped_acquire_simple &operator=(const gil_scoped_acquire_simple &) = delete;

    /// The destructor will not call PyGILState_Release, so the GIL stays held.
    /// Use this if the interpreter could be shutting down when the scope ends.
    PYBIND11_NOINLINE void disarm() { active = false; }

    ~gil_scoped_acquire_simple() {
        if (active) {
            PyGILState_Release(state);
        }
    }
};

class gil_scoped_release_simple {
    PyThreadState *state;
    bool active = true;

public:
    // PRECONDITION: The GIL must be held when this constructor is called.
    gil_scoped_release_simple() {
#if !defined(Py_LIMITED_API) // PyGILState_Check is not part of the stable ABI
        assert(PyGILState_Check());
#endif
        state = PyEval_SaveThread();
    }
    gil_scoped_release_simple(const gil_scoped_release_simple &) = delete;
    gil_scoped_release_simple &operator=(const gil_scoped_release_simple &) = delete;

    /// The destructor will not call PyEval_RestoreThread, so the GIL is not
    /// acquired again. Use this if the interpreter could be shutting down when
    /// the scope ends.
    PYBIND11_NOINLINE void disarm() { active = false; }

    ~gil_scoped_release_simple() {
        if (active) {
            PyEval_RestoreThread(state);
        }
    }
};

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
