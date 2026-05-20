/*
    pybind11/subinterpreter.h: Support for creating and using subinterpreters

    Copyright (c) 2025 The Pybind Development Team.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "detail/common.h"
#include "detail/internals.h"
#include "gil.h"

#include <stdexcept>

#ifndef PYBIND11_HAS_SUBINTERPRETER_SUPPORT
#    error "This platform does not support subinterpreters, do not include this file."
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

class subinterpreter;
class subinterpreter_thread_state;

/// Activate the subinterpreter and acquire its GIL, while also releasing any GIL and interpreter
/// currently held. Upon exiting the scope, the previous subinterpreter (if any) and its
/// associated GIL are restored to their state as they were before the scope was entered.
///
/// Two construction modes are supported:
///
/// 1. `subinterpreter_scoped_activate(subinterpreter const &)`:
///    Transient mode (the default).  A fresh PyThreadState is created on entry and destroyed on
///    exit.  This is the established behavior; existing code is unaffected.
///
/// 2. `subinterpreter_scoped_activate(subinterpreter_thread_state &)`:
///    Reuse mode.  The PyThreadState owned by the given subinterpreter_thread_state is swapped
///    in on entry and swapped out (but NOT destroyed) on exit, so repeated activations on the
///    same OS thread reuse the same PyThreadState and preserve its per-thread interpreter state.
///    Use this when a single OS thread re-enters one or more subinterpreters many times.
class subinterpreter_scoped_activate {
public:
    explicit subinterpreter_scoped_activate(subinterpreter const &si);
    explicit subinterpreter_scoped_activate(subinterpreter_thread_state &ts);
    ~subinterpreter_scoped_activate();

    subinterpreter_scoped_activate(subinterpreter_scoped_activate &&) = delete;
    subinterpreter_scoped_activate(subinterpreter_scoped_activate const &) = delete;
    subinterpreter_scoped_activate &operator=(subinterpreter_scoped_activate &) = delete;
    subinterpreter_scoped_activate &operator=(subinterpreter_scoped_activate const &) = delete;

private:
    PyThreadState *old_tstate_ = nullptr;
    PyThreadState *tstate_ = nullptr;
    PyGILState_STATE gil_state_;
    bool simple_gil_ = false;
    // When true, tstate_ is owned by a subinterpreter_thread_state and must NOT be destroyed
    // when this scope exits (only swapped out).
    bool borrowed_ = false;
};

/// Holds a Python subinterpreter instance
class subinterpreter {
public:
    /// empty/unusable, but move-assignable.  use create() to create a subinterpreter.
    subinterpreter() = default;

    subinterpreter(subinterpreter const &copy) = delete;
    subinterpreter &operator=(subinterpreter const &copy) = delete;

    subinterpreter(subinterpreter &&old) noexcept
        : istate_(old.istate_), creation_tstate_(old.creation_tstate_) {
        old.istate_ = nullptr;
        old.creation_tstate_ = nullptr;
    }

    subinterpreter &operator=(subinterpreter &&old) noexcept {
        std::swap(old.istate_, istate_);
        std::swap(old.creation_tstate_, creation_tstate_);
        return *this;
    }

    /// Create a new subinterpreter with the specified configuration
    /// @note This function acquires (and then releases) the main interpreter GIL, but the main
    /// interpreter and its GIL are not required to be held prior to calling this function.
    static subinterpreter create(PyInterpreterConfig const &cfg) {

        error_scope err_scope;
        subinterpreter result;
        {
            // we must hold the main GIL in order to create a subinterpreter
            subinterpreter_scoped_activate main_guard(main());

            auto *prev_tstate = PyThreadState_Get();

            PyStatus status;

            {
                /*
                Several internal CPython modules are lacking proper subinterpreter support in 3.12
                even though it is "stable" in that version.  This most commonly seems to cause
                crashes when two interpreters concurrently initialize, which imports several things
                (like builtins, unicode, codecs).
                */
#if PY_VERSION_HEX < 0x030D0000 && defined(Py_MOD_PER_INTERPRETER_GIL_SUPPORTED)
                static std::mutex one_at_a_time;
                std::lock_guard<std::mutex> guard(one_at_a_time);
#endif
                status = Py_NewInterpreterFromConfig(&result.creation_tstate_, &cfg);
            }

            // this doesn't raise a normal Python exception, it provides an exit() status code.
            if (PyStatus_Exception(status) != 0) {
                pybind11_fail("failed to create new sub-interpreter");
            }

            // upon success, the new interpreter is activated in this thread
            result.istate_ = result.creation_tstate_->interp;
            detail::has_seen_non_main_interpreter() = true;
            detail::get_internals(); // initialize internals.tstate, amongst other things...

            // In 3.13+ this state should be deleted right away, and the memory will be reused for
            // the next threadstate on this interpreter. However, on 3.12 we cannot do that, we
            // must keep it around (but not use it) ... see destructor.
#if PY_VERSION_HEX >= 0x030D0000
            PyThreadState_Clear(result.creation_tstate_);
            PyThreadState_DeleteCurrent();
#endif

            // we have to switch back to main, and then the scopes will handle cleanup
            PyThreadState_Swap(prev_tstate);
        }
        return result;
    }

    /// Calls create() with a default configuration of an isolated interpreter that disallows fork,
    /// exec, and Python threads.
    static subinterpreter create() {
        // same as the default config in the python docs
        PyInterpreterConfig cfg;
        std::memset(&cfg, 0, sizeof(cfg));
        cfg.allow_threads = 1;
        cfg.check_multi_interp_extensions = 1;
        cfg.gil = PyInterpreterConfig_OWN_GIL;
        return create(cfg);
    }

    ~subinterpreter() {
        if (!creation_tstate_) {
            // non-owning wrapper, do nothing.
            return;
        }

        PyThreadState *destroy_tstate = nullptr;
        PyThreadState *old_tstate = nullptr;

        // Python 3.12 requires us to keep the original PyThreadState alive until we are ready to
        // destroy the interpreter.  We prefer to use that to destroy the interpreter.
#if PY_VERSION_HEX < 0x030D0000
        // The tstate passed to Py_EndInterpreter MUST have been created on the current OS thread.
        bool same_thread = false;
#    ifdef PY_HAVE_THREAD_NATIVE_ID
        same_thread = PyThread_get_thread_native_id() == creation_tstate_->native_thread_id;
#    endif
        if (same_thread) {
            // OK it is safe to use the creation state here
            destroy_tstate = creation_tstate_;
            old_tstate = PyThreadState_Swap(destroy_tstate);
        } else {
            // We have to make a new tstate on this thread and use that.
            destroy_tstate = PyThreadState_New(istate_);
            old_tstate = PyThreadState_Swap(destroy_tstate);

            // We can use the one we just created, so we must delete the creation state.
            PyThreadState_Clear(creation_tstate_);
            PyThreadState_Delete(creation_tstate_);
        }
#else
        destroy_tstate = PyThreadState_New(istate_);
        old_tstate = PyThreadState_Swap(destroy_tstate);
#endif

        bool switch_back = (old_tstate != nullptr) && old_tstate->interp != istate_;

        // Internals always exists in the subinterpreter, this class enforces it when it creates
        // the subinterpreter. Even if it didn't, this only creates the pointer-to-pointer, not the
        // internals themselves.
        detail::get_internals_pp_manager().get_pp();
        detail::get_local_internals_pp_manager().get_pp();

        // End it
        Py_EndInterpreter(destroy_tstate);

        // It's possible for the  internals to be created during endinterpreter (e.g. if a
        // py::capsule calls `get_internals()` during destruction), so we destroy afterward.
        detail::get_internals_pp_manager().destroy();
        detail::get_local_internals_pp_manager().destroy();

        // switch back to the old tstate and old GIL (if there was one)
        if (switch_back) {
            PyThreadState_Swap(old_tstate);
        }
    }

    /// Get a handle to the main interpreter that can be used with subinterpreter_scoped_activate
    /// Note that destructing the handle is a noop, the main interpreter can only be ended by
    /// py::finalize_interpreter()
    static subinterpreter main() {
        subinterpreter m;
        m.istate_ = PyInterpreterState_Main();
        m.disarm(); // make destruct a noop
        return m;
    }

    /// Get a non-owning wrapper of the currently active interpreter (if any)
    static subinterpreter current() {
        subinterpreter c;
        c.istate_ = detail::get_interpreter_state_unchecked();
        c.disarm(); // make destruct a noop, we don't own this...
        return c;
    }

    /// Get the numerical identifier for the sub-interpreter
    int64_t id() const {
        if (istate_ != nullptr) {
            return PyInterpreterState_GetID(istate_);
        }
        return -1; // CPython uses one-up numbers from 0, so negative should be safe to return
                   // here.
    }

    /// Get the interpreter's state dict.  This interpreter's GIL must be held before calling!
    dict state_dict() { return reinterpret_borrow<dict>(PyInterpreterState_GetDict(istate_)); }

    /// abandon cleanup of this subinterpreter (leak it). this might be needed during
    /// finalization...
    void disarm() { creation_tstate_ = nullptr; }

    /// An empty wrapper cannot be activated
    bool empty() const { return istate_ == nullptr; }

    /// Is this wrapper non-empty
    explicit operator bool() const { return !empty(); }

private:
    friend class subinterpreter_scoped_activate;
    friend class subinterpreter_thread_state;
    PyInterpreterState *istate_ = nullptr;
    PyThreadState *creation_tstate_ = nullptr;
};

/// RAII wrapper that owns a PyThreadState bound to a specific subinterpreter on the OS thread
/// that constructed it.  Intended to be held long-lived (e.g. as a `thread_local`, or inside a
/// per-thread struct) so that many subinterpreter_scoped_activate scopes on the same OS thread
/// can reuse a single PyThreadState instead of creating and destroying one each time.
///
/// The PyThreadState is created on construction in a *released* state: it is NOT made current,
/// and no GIL is acquired.  Activation is the job of subinterpreter_scoped_activate.
///
/// A single OS thread can hold one of these per subinterpreter and alternate between them via
/// subinterpreter_scoped_activate without churning PyThreadState objects.
///
/// Lifetime / threading requirements:
///
/// - Construction and destruction must happen on the SAME OS thread (a PyThreadState is bound
///   to the OS thread that created it; deleting it on a different thread is undefined behavior).
/// - The owning subinterpreter must still be alive when this object is destroyed.
/// - This object must NOT be destroyed while a subinterpreter_scoped_activate referring to it is
///   still alive (the activator holds a reference into it).
///
/// Typical usage:
///
/// @code
///   thread_local py::subinterpreter_thread_state ts(sub);
///   {
///       py::subinterpreter_scoped_activate guard(ts);   // swap-in only
///       // ... use the subinterpreter ...
///   }                                                    // swap-out, tstate kept alive
///   {
///       py::subinterpreter_scoped_activate guard(ts);   // reuses the same PyThreadState
///       // ...
///   }
/// @endcode
class subinterpreter_thread_state {
public:
    /// Create a PyThreadState for `si` on the calling OS thread.  The new state is left in a
    /// released state (not current, no GIL acquired).
    explicit subinterpreter_thread_state(subinterpreter const &si);

    /// Destroy the owned PyThreadState.  Must run on the same OS thread that constructed this
    /// object, while the owning subinterpreter is still alive, and while no
    /// subinterpreter_scoped_activate referring to this object is alive.
    ~subinterpreter_thread_state();

    subinterpreter_thread_state(subinterpreter_thread_state const &) = delete;
    subinterpreter_thread_state(subinterpreter_thread_state &&) = delete;
    subinterpreter_thread_state &operator=(subinterpreter_thread_state const &) = delete;
    subinterpreter_thread_state &operator=(subinterpreter_thread_state &&) = delete;

    /// The interpreter this thread state belongs to.
    PyInterpreterState *interpreter_state() const { return istate_; }

    /// The owned PyThreadState pointer; valid for the lifetime of this object.
    PyThreadState *raw_thread_state() const { return tstate_; }

private:
    friend class subinterpreter_scoped_activate;
    PyThreadState *tstate_ = nullptr;
    PyInterpreterState *istate_ = nullptr;
};

class scoped_subinterpreter {
public:
    scoped_subinterpreter() : si_(subinterpreter::create()), scope_(si_) {}

    explicit scoped_subinterpreter(PyInterpreterConfig const &cfg)
        : si_(subinterpreter::create(cfg)), scope_(si_) {}

private:
    subinterpreter si_;
    subinterpreter_scoped_activate scope_;
};

// --- subinterpreter_scoped_activate -----------------------------------------------------------

inline subinterpreter_scoped_activate::subinterpreter_scoped_activate(subinterpreter const &si) {
    if (!si.istate_) {
        pybind11_fail("null subinterpreter");
    }

    if (detail::get_interpreter_state_unchecked() == si.istate_) {
        // we are already on this interpreter, make sure we hold the GIL
        simple_gil_ = true;
        gil_state_ = PyGILState_Ensure();
        return;
    }

    // we can't really interact with the interpreter at all until we switch to it
    // not even to, for example, look in its state dict or touch its internals
    tstate_ = PyThreadState_New(si.istate_);

    // make the interpreter active and acquire the GIL
    old_tstate_ = PyThreadState_Swap(tstate_);

    // save this in internals for scoped_gil calls (see also: PR #5870)
    detail::get_internals().tstate = tstate_;
}

inline subinterpreter_scoped_activate::subinterpreter_scoped_activate(
    subinterpreter_thread_state &ts) {
    if (ts.tstate_ == nullptr) {
        pybind11_fail("subinterpreter_scoped_activate: empty subinterpreter_thread_state");
    }

    if (detail::get_interpreter_state_unchecked() == ts.istate_) {
        // We are already on this interpreter -- e.g. nested activation, or a different
        // PyThreadState for the same interpreter is already current on this thread.  Match the
        // fast path of the (subinterpreter const&) overload: just ensure the GIL is held.  The
        // `ts` argument's PyThreadState is intentionally NOT swapped to here; the already-current
        // tstate keeps being used until the outer scope exits.
        simple_gil_ = true;
        gil_state_ = PyGILState_Ensure();
        return;
    }

    tstate_ = ts.tstate_;
    borrowed_ = true;

    // make the interpreter active and acquire the GIL
    old_tstate_ = PyThreadState_Swap(tstate_);

    // save this in internals for scoped_gil calls (see also: PR #5870)
    detail::get_internals().tstate = tstate_;
}

inline subinterpreter_scoped_activate::~subinterpreter_scoped_activate() {
    if (simple_gil_) {
        // We were on this interpreter already, so just make sure the GIL goes back as it was
        PyGILState_Release(gil_state_);
    } else {
        if (tstate_) {
#if defined(PYBIND11_DETAILED_ERROR_MESSAGES)
            if (detail::get_thread_state_unchecked() != tstate_) {
                pybind11_fail("~subinterpreter_scoped_activate: thread state must be current!");
            }
#endif
            detail::get_internals().tstate.reset();
            if (!borrowed_) {
                PyThreadState_Clear(tstate_);
                PyThreadState_DeleteCurrent();
            }
            // When borrowed_, tstate_ stays alive in its owning subinterpreter_thread_state for
            // reuse; the PyThreadState_Swap below merely detaches it from this thread.
        }

        // Go back the previous interpreter (if any) and acquire THAT gil
        PyThreadState_Swap(old_tstate_);
    }
}

// --- subinterpreter_thread_state --------------------------------------------------------------

inline subinterpreter_thread_state::subinterpreter_thread_state(subinterpreter const &si) {
    if (!si.istate_) {
        pybind11_fail("subinterpreter_thread_state: null subinterpreter");
    }
    istate_ = si.istate_;
    // PyThreadState_New does not require holding any GIL and does not make the new state current.
    tstate_ = PyThreadState_New(istate_);
    if (tstate_ == nullptr) {
        pybind11_fail("subinterpreter_thread_state: PyThreadState_New returned null");
    }
}

inline subinterpreter_thread_state::~subinterpreter_thread_state() {
    if (tstate_ == nullptr) {
        return;
    }
    // The PyThreadState must be made current to be cleared and deleted on the owning OS thread.
    // Swap it in (which acquires the subinterpreter's GIL), clear+delete, then restore whatever
    // was active before.
    PyThreadState *prev = PyThreadState_Swap(tstate_);
    PyThreadState_Clear(tstate_);
    PyThreadState_DeleteCurrent();
    // If `prev` is tstate_ itself, the user destroyed this object while it was active via a
    // subinterpreter_scoped_activate -- a contract violation, but be defensive: do NOT swap back
    // to a now-deleted pointer.  Leaving the thread with no current interpreter is consistent
    // with the cached state having just been destroyed.
    if (prev != nullptr && prev != tstate_) {
        PyThreadState_Swap(prev);
    }
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
