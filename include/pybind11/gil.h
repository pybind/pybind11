/*
    pybind11/gil.h: RAII helpers for managing the GIL

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

// common.h must come first: on PyPy/GraalPy it defines PYBIND11_SIMPLE_GIL_MANAGEMENT,
// which selects the branch below.
#include "detail/common.h"

#if defined(PYBIND11_SIMPLE_GIL_MANAGEMENT)

#    include "gil_simple.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

using gil_scoped_acquire = gil_scoped_acquire_simple;
using gil_scoped_release = gil_scoped_release_simple;

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

#else

#    include "detail/internals.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

PYBIND11_NAMESPACE_BEGIN(detail)

PYBIND11_WARNING_PUSH
PYBIND11_WARNING_DISABLE_GCC("-Wredundant-decls")

// forward declarations
PyThreadState *get_thread_state_unchecked();

PYBIND11_WARNING_POP

PYBIND11_NAMESPACE_END(detail)

/* The functions below essentially reproduce the PyGILState_* API using a RAII
 * pattern, but there are a few important differences:
 *
 * 1. When acquiring the GIL from an non-main thread during the finalization
 *    phase, the GILState API blindly terminates the calling thread, which
 *    is often not what is wanted. This API does not do this.
 *
 * 2. The gil_scoped_release function can optionally cut the relationship
 *    of a PyThreadState and its associated thread, which allows moving it to
 *    another thread (this is a fairly rare/advanced use case).
 *
 * 3. The reference count of an acquired thread state can be controlled. This
 *    can be handy to prevent cases where callbacks issued from an external
 *    thread would otherwise constantly construct and destroy thread state data
 *    structures.
 *
 * See the Python bindings of NanoGUI (http://github.com/wjakob/nanogui) for an
 * example which uses features 2 and 3 to migrate the Python thread of
 * execution to another thread (to run the event loop on the original thread,
 * in this case).
 */

class gil_scoped_acquire {
public:
    gil_scoped_acquire();

    gil_scoped_acquire(const gil_scoped_acquire &) = delete;
    gil_scoped_acquire &operator=(const gil_scoped_acquire &) = delete;

    void inc_ref() { ++tstate->gilstate_counter; }

    void dec_ref();

    /// This method will disable the PyThreadState_DeleteCurrent call and the
    /// GIL won't be released. This method should be used if the interpreter
    /// could be shutting down when this is called, as thread deletion is not
    /// allowed during shutdown. Check _Py_IsFinalizing() on Python 3.7+, and
    /// protect subsequent code.
    PYBIND11_NOINLINE void disarm() { active = false; }

    ~gil_scoped_acquire();

private:
    PyThreadState *tstate = nullptr;
    bool release = true;
    bool active = true;
};

class gil_scoped_release {
public:
    // PRECONDITION: The GIL must be held when this constructor is called.
    explicit gil_scoped_release(bool disassoc = false);

    gil_scoped_release(const gil_scoped_release &) = delete;
    gil_scoped_release &operator=(const gil_scoped_release &) = delete;

    /// This method will disable the PyThreadState_DeleteCurrent call and the
    /// GIL won't be acquired. This method should be used if the interpreter
    /// could be shutting down when this is called, as thread deletion is not
    /// allowed during shutdown. Check _Py_IsFinalizing() on Python 3.7+, and
    /// protect subsequent code.
    PYBIND11_NOINLINE void disarm() { active = false; }

    ~gil_scoped_release();

private:
    PyThreadState *tstate;
    bool disassoc;
    bool active = true;
};

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

#    ifndef PYBIND11_PRECOMPILED
#        include "gil-inl.h" // IWYU pragma: export
#    endif

#endif // !PYBIND11_SIMPLE_GIL_MANAGEMENT
