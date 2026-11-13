/*
    pybind11/gil-inl.h: Out-of-line definitions for gil.h

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

// Every function defined here must start with PYBIND11_INLINE (or
// PYBIND11_NOINLINE_ATTR PYBIND11_INLINE). In the default header-only mode this file is
// included at the bottom of gil.h; when PYBIND11_PRECOMPILED is defined it is only
// compiled into the pybind11 static library (see src/).

#pragma once

#include "gil.h"

#if !defined(PYBIND11_SIMPLE_GIL_MANAGEMENT)

#    include <cassert>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

PYBIND11_NOINLINE_ATTR PYBIND11_INLINE gil_scoped_acquire::gil_scoped_acquire() {
    auto &internals = detail::get_internals();
    tstate = internals.tstate.get();

    if (!tstate) {
        /* Check if the GIL was acquired using the PyGILState_* API instead (e.g. if
           calling from a Python thread). Since we use a different key, this ensures
           we don't create a new thread state and deadlock in PyEval_AcquireThread
           below. Note we don't save this state with internals.tstate, since we don't
           create it we would fail to clear it (its reference count should be > 0). */
        tstate = PyGILState_GetThisThreadState();
    }

    if (!tstate) {
        tstate = PyThreadState_New(internals.istate);
#    if defined(PYBIND11_DETAILED_ERROR_MESSAGES)
        if (!tstate) {
            pybind11_fail("scoped_acquire: could not create thread state!");
        }
#    endif
        tstate->gilstate_counter = 0;
        internals.tstate = tstate;
    } else {
        release = detail::get_thread_state_unchecked() != tstate;
    }

    if (release) {
        PyEval_AcquireThread(tstate);
    }

    inc_ref();
}

PYBIND11_NOINLINE_ATTR PYBIND11_INLINE void gil_scoped_acquire::dec_ref() {
    --tstate->gilstate_counter;
#    if defined(PYBIND11_DETAILED_ERROR_MESSAGES)
    if (detail::get_thread_state_unchecked() != tstate) {
        pybind11_fail("scoped_acquire::dec_ref(): thread state must be current!");
    }
    if (tstate->gilstate_counter < 0) {
        pybind11_fail("scoped_acquire::dec_ref(): reference count underflow!");
    }
#    endif
    if (tstate->gilstate_counter == 0) {
#    if defined(PYBIND11_DETAILED_ERROR_MESSAGES)
        if (!release) {
            pybind11_fail("scoped_acquire::dec_ref(): internal error!");
        }
#    endif
        // Make sure that PyThreadState_Clear is not recursively called by finalizers.
        // See issue #5827
        ++tstate->gilstate_counter;
        PyThreadState_Clear(tstate);
        --tstate->gilstate_counter;
        if (active) {
            PyThreadState_DeleteCurrent();
        }
        detail::get_internals().tstate.reset();
        release = false;
    }
}

PYBIND11_NOINLINE_ATTR PYBIND11_INLINE gil_scoped_acquire::~gil_scoped_acquire() {
    dec_ref();
    if (release) {
        PyEval_SaveThread();
    }
}

PYBIND11_INLINE gil_scoped_release::gil_scoped_release(bool disassoc) : disassoc(disassoc) {
    assert(PyGILState_Check());
    // `get_internals()` must be called here unconditionally in order to initialize
    // `internals.tstate` for subsequent `gil_scoped_acquire` calls. Otherwise, an
    // initialization race could occur as multiple threads try `gil_scoped_acquire`.
    auto &internals = detail::get_internals();
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    tstate = PyEval_SaveThread();
    if (disassoc) {
        internals.tstate.reset();
    }
}

PYBIND11_INLINE gil_scoped_release::~gil_scoped_release() {
    if (!tstate) {
        return;
    }
    // `PyEval_RestoreThread()` should not be called if runtime is finalizing
    if (active) {
        PyEval_RestoreThread(tstate);
    }
    if (disassoc) {
        detail::get_internals().tstate = tstate;
    }
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

#endif // !PYBIND11_SIMPLE_GIL_MANAGEMENT
