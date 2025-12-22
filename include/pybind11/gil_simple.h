// Copyright (c) 2016-2025 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include "detail/common.h"

#include <cassert>
#ifdef Py_GIL_DISABLED
#    include <mutex>
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

#ifdef Py_GIL_DISABLED
namespace detail {

// Compatibility mutex for free-threaded Python builds.
// In traditional Python, the GIL provides mutual exclusion for code that acquires it.
// In free-threaded Python, there is no GIL, so existing code that assumes mutual exclusion
// after gil_scoped_acquire would have data races. This mutex restores that safety guarantee.
//
// This is intentionally a global mutex (not per-interpreter) for simplicity. The performance
// cost is acceptable as a safe default; code that needs maximum parallelism can be migrated
// to use explicit locking or the lighter-weight scoped_ensure_thread_state helper.
inline std::mutex &get_compat_mutex() {
    static std::mutex mtx;
    return mtx;
}

// Thread-local flag to track whether this thread holds the compat mutex.
// This is needed because the main thread starts with Python initialized (holding the "GIL")
// but we don't lock the compat mutex at that point. We only want to lock/unlock when
// transitioning via gil_scoped_acquire/release.
inline bool &compat_mutex_held_by_this_thread() {
    static thread_local bool held = false;
    return held;
}

inline void acquire_compat_mutex() {
    if (!compat_mutex_held_by_this_thread()) {
        get_compat_mutex().lock();
        compat_mutex_held_by_this_thread() = true;
    }
}

inline void release_compat_mutex() {
    if (compat_mutex_held_by_this_thread()) {
        compat_mutex_held_by_this_thread() = false;
        get_compat_mutex().unlock();
    }
}

} // namespace detail
#endif

class gil_scoped_acquire_simple {
    PyGILState_STATE state;
#ifdef Py_GIL_DISABLED
    bool acquired_compat_mutex_ = false;
#endif

public:
    gil_scoped_acquire_simple() : state{PyGILState_Ensure()} {
#ifdef Py_GIL_DISABLED
        if (!detail::compat_mutex_held_by_this_thread()) {
            detail::get_compat_mutex().lock();
            detail::compat_mutex_held_by_this_thread() = true;
            acquired_compat_mutex_ = true;
        }
#endif
    }
    gil_scoped_acquire_simple(const gil_scoped_acquire_simple &) = delete;
    gil_scoped_acquire_simple &operator=(const gil_scoped_acquire_simple &) = delete;
    ~gil_scoped_acquire_simple() {
#ifdef Py_GIL_DISABLED
        if (acquired_compat_mutex_) {
            detail::compat_mutex_held_by_this_thread() = false;
            detail::get_compat_mutex().unlock();
        }
#endif
        PyGILState_Release(state);
    }
};

class gil_scoped_release_simple {
    PyThreadState *state;
#ifdef Py_GIL_DISABLED
    bool released_compat_mutex_ = false;
#endif

public:
    // PRECONDITION: The GIL must be held when this constructor is called.
    gil_scoped_release_simple() {
        assert(PyGILState_Check());
#ifdef Py_GIL_DISABLED
        if (detail::compat_mutex_held_by_this_thread()) {
            detail::compat_mutex_held_by_this_thread() = false;
            detail::get_compat_mutex().unlock();
            released_compat_mutex_ = true;
        }
#endif
        state = PyEval_SaveThread();
    }
    gil_scoped_release_simple(const gil_scoped_release_simple &) = delete;
    gil_scoped_release_simple &operator=(const gil_scoped_release_simple &) = delete;
    ~gil_scoped_release_simple() {
        PyEval_RestoreThread(state);
#ifdef Py_GIL_DISABLED
        if (released_compat_mutex_) {
            detail::get_compat_mutex().lock();
            detail::compat_mutex_held_by_this_thread() = true;
        }
#endif
    }
};

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
