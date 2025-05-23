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

#if !defined(PYBIND11_HAS_SUBINTERPRETER_SUPPORT)
#    error "This platform does not support subinterpreters, do not include this file."
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)
PyInterpreterState *get_interpreter_state_unchecked() {
    auto cur_tstate = get_thread_state_unchecked();
    if (cur_tstate)
        return cur_tstate->interp;
    else
        return nullptr;
}
PYBIND11_NAMESPACE_END(detail)

class subinterpreter;

/// Activate the subinterpreter and acquire its GIL, while also releasing any GIL and interpreter
/// currently held. Upon exiting the scope, the previous subinterpreter (if any) and its
/// associated GIL are restored to their state as they were before the scope was entered.
class subinterpreter_scoped_activate {
public:
    explicit subinterpreter_scoped_activate(subinterpreter const &si);
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
    static inline subinterpreter create(PyInterpreterConfig const &cfg) {
        error_scope err_scope;
        subinterpreter result;
        {
            // we must hold the main GIL in order to create a subinterpreter
            subinterpreter_scoped_activate main_guard(main());

            auto prev_tstate = PyThreadState_Get();

            auto status = Py_NewInterpreterFromConfig(&result.creation_tstate_, &cfg);

            // this doesn't raise a normal Python exception, it provides an exit() status code.
            if (PyStatus_Exception(status)) {
                pybind11_fail("failed to create new sub-interpreter");
            }

            // upon success, the new interpreter is activated in this thread
            result.istate_ = result.creation_tstate_->interp;
            detail::get_num_interpreters_seen() += 1; // there are now many interpreters
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
    static inline subinterpreter create() {
        // same as the default config in the python docs
        PyInterpreterConfig cfg;
        std::memset(&cfg, 0, sizeof(cfg));
        cfg.check_multi_interp_extensions = 1;
        cfg.gil = PyInterpreterConfig_OWN_GIL;
        return create(cfg);
    }

    ~subinterpreter() {
        if (!creation_tstate_) {
            // non-owning wrapper, do nothing.
            return;
        }

        PyThreadState *destroy_tstate;
        PyThreadState *old_tstate;

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

        bool switch_back = old_tstate && old_tstate->interp != istate_;

        // Get the internals pointer (without creating it if it doesn't exist).  It's possible
        // for the internals to be created during Py_EndInterpreter() (e.g. if a py::capsule
        // calls `get_internals()` during destruction), so we get the pointer-pointer here and
        // check it after.
        auto *&internals_ptr_ptr = detail::get_internals_pp<detail::internals>();
        auto *&local_internals_ptr_ptr = detail::get_internals_pp<detail::local_internals>();
        {
            dict sd = state_dict();
            internals_ptr_ptr
                = detail::get_internals_pp_from_capsule_in_state_dict<detail::internals>(
                    sd, PYBIND11_INTERNALS_ID);
            local_internals_ptr_ptr
                = detail::get_internals_pp_from_capsule_in_state_dict<detail::local_internals>(
                    sd, detail::get_local_internals_id());
        }

        // End it
        Py_EndInterpreter(destroy_tstate);

        // do NOT decrease detail::get_num_interpreters_seen, because it can never decrease
        // while other threads are running...

        if (internals_ptr_ptr) {
            internals_ptr_ptr->reset();
        }
        if (local_internals_ptr_ptr) {
            local_internals_ptr_ptr->reset();
        }

        // switch back to the old tstate and old GIL (if there was one)
        if (switch_back)
            PyThreadState_Swap(old_tstate);
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
        if (istate_ != nullptr)
            return PyInterpreterState_GetID(istate_);
        else
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
    PyInterpreterState *istate_ = nullptr;
    PyThreadState *creation_tstate_ = nullptr;
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

    // save this in internals for scoped_gil calls
    PYBIND11_TLS_REPLACE_VALUE(detail::get_internals().tstate, tstate_);
}

inline subinterpreter_scoped_activate::~subinterpreter_scoped_activate() {
    if (simple_gil_) {
        // We were on this interpreter already, so just make sure the GIL goes back as it was
        PyGILState_Release(gil_state_);
    } else {
#if defined(PYBIND11_DETAILED_ERROR_MESSAGES)
        bool has_active_exception;
#    if defined(__cpp_lib_uncaught_exceptions)
        has_active_exception = std::uncaught_exceptions() > 0;
#    else
        // removed in C++20, replaced with uncaught_exceptions
        has_active_exception = std::uncaught_exception();
#    endif
        if (has_active_exception) {
            try {
                std::rethrow_exception(std::current_exception());
            } catch (error_already_set &) {
                // Because error_already_set holds python objects and what() acquires the GIL, it
                // is basically never OK to let these exceptions propagate outside the current
                // active interpreter.
                pybind11_fail("~subinterpreter_scoped_activate: cannot propagate Python "
                              "exceptions outside of their owning interpreter");
            } catch (...) {
            }
        }
#endif

        if (tstate_) {
#if defined(PYBIND11_DETAILED_ERROR_MESSAGES)
            if (detail::get_thread_state_unchecked() != tstate_) {
                pybind11_fail("~subinterpreter_scoped_activate: thread state must be current!");
            }
#endif
            PYBIND11_TLS_DELETE_VALUE(detail::get_internals().tstate);
            PyThreadState_Clear(tstate_);
            PyThreadState_DeleteCurrent();
        }

        // Go back the previous interpreter (if any) and acquire THAT gil
        PyThreadState_Swap(old_tstate_);
    }
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
