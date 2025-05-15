#pragma once

#include "detail/common.h"
#include "detail/internals.h"
#include "gil.h"

#include <stdexcept>

#if !defined(PYBIND11_SUBINTERPRETER_SUPPORT)
#    error "This platform does not support subinterpreters, do not include this file."
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

class subinterpreter;

/// Activate the subinterpreter and acquire it's GIL, while also releasing any GIL and interpreter
/// currently held. Upon exiting the scope, the previous subinterpreter (if any) and it's
/// associated GIL are restored to their state as they were before the scope was entered.
class subinterpreter_scoped_activate {
public:
    explicit subinterpreter_scoped_activate(subinterpreter const &si);
    ~subinterpreter_scoped_activate();

private:
    PyThreadState *old_tstate_ = nullptr;
    PyThreadState *free_tstate_ = nullptr;
    PyGILState_STATE gil_state_;
    bool simple_gil_ = false;
};

class subinterpreter {
public:
    subinterpreter() = default;
    subinterpreter(subinterpreter const &copy) = delete;
    subinterpreter &operator=(subinterpreter const &copy) = delete;

    subinterpreter(subinterpreter &&old) : tstate_(old.tstate_), istate_(old.istate_) {
        old.tstate_ = nullptr;
        old.istate_ = nullptr;
    }

    subinterpreter &operator=(subinterpreter &&old) {
        std::swap(old.tstate_, tstate_);
        std::swap(old.istate_, istate_);
        return *this;
    }

    ~subinterpreter() {
        if (tstate_) {
            if (PyThread_get_thread_ident() != tstate_->native_thread_id) {
                // Throwing from destructors is bad :(
                // But if we don't throw, we either leak the interpreter or the code hangs because
                // internal Python TSS values are wrong/missing
                throw std::runtime_error(
                    "wrong thread called subinterpreter destruct. subinterpreters can only "
                    "destruct on the thread that created them!");
            }

            // has to be the active interpreter in order to call End on it
            // switch into the expiring interpreter
            auto old_tstate = PyThreadState_Swap(tstate_);

            // make sure we have the GIL
            (void) PyGILState_Ensure();

            // End it
            Py_EndInterpreter(tstate_);

            // switch back to the old tstate and old GIL (if there was one)
            if (old_tstate != tstate_)
                PyThreadState_Swap(old_tstate);

            // do NOT decrease detail::get_num_interpreters_seen, because it can never decrease
            // while other threads are running...
        }
    }

    /// abandon cleanup of this subinterpreter.  might be needed during finalization
    void disarm() { tstate_ = nullptr; }

    /// Get a handle to the main interpreter that can be used with subinterpreter_scoped_activate
    /// Note that destructing the handle is a noop, the main interpreter can only be ended by
    /// py::finalize_interpreter()
    static subinterpreter_scoped_activate main_scoped_activate() {
        subinterpreter m;
        m.istate_ = PyInterpreterState_Main();
        m.disarm(); // make destruct a noop
        return subinterpreter_scoped_activate(m);
    }

    /// Create a new subinterpreter with the specified configuration
    /// Note Well:
    static inline subinterpreter create(PyInterpreterConfig const &cfg) {
        error_scope err_scope;
        auto main_guard = main_scoped_activate();
        subinterpreter result;
        {
            // we must hold the main GIL in order to create a subinterpreter
            gil_scoped_acquire gil;

            auto prev_tstate = PyThreadState_Get();

            auto status = Py_NewInterpreterFromConfig(&result.tstate_, &cfg);

            // this doesn't raise a normal Python exception, it provides an exit() status code.
            if (PyStatus_Exception(status)) {
                pybind11_fail("failed to create new sub-interpreter");
            }

            // upon success, the new interpreter is activated in this thread
            result.istate_ = result.tstate_->interp;
            detail::get_num_interpreters_seen() += 1; // there are now many interpreters
            detail::get_internals(); // initialize internals.tstate, amongst other things...

            // we have to switch back to main, and then the scopes will handle cleanup
            PyThreadState_Swap(prev_tstate);
        }
        return result;
    }

    /// Call create() with a default configuration of an isolated interpreter that disallows fork,
    /// exec, and Python threads.
    static inline subinterpreter create() {
        // same as the default config in the python docs
        PyInterpreterConfig cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.check_multi_interp_extensions = 1;
        cfg.gil = PyInterpreterConfig_OWN_GIL;
        return create(cfg);
    }

private:
    friend class subinterpreter_scoped_activate;
    PyThreadState *tstate_ = nullptr;
    PyInterpreterState *istate_ = nullptr;
};

subinterpreter_scoped_activate::subinterpreter_scoped_activate(subinterpreter const &si) {
#if defined(PYBIND11_DETAILED_ERROR_MESSAGES)
    if (!si.istate_ || !si.tstate_) {
        pybind11_fail("null subinterpreter");
    }
#endif

    auto cur_tstate = detail::get_thread_state_unchecked();
    if (cur_tstate && cur_tstate->interp == si.istate_) {
        // we are already on this interpreter, make sure we hold the GIL
        simple_gil_ = true;
        gil_state_ = PyGILState_Ensure();
        return;
    }

    PyThreadState *desired_tstate = nullptr;

    // get the state dict for the interpreter we want
    dict idict = reinterpret_borrow<dict>(PyInterpreterState_GetDict(si.istate_));
    // and get the internals from it
    auto *internals_pp = detail::get_internals_pp_from_capsule_in_state_dict<detail::internals>(
        idict, PYBIND11_INTERNALS_ID);
    if (internals_pp && *internals_pp) {
        // see if there is already a tstate for this thread
        desired_tstate = (PyThreadState *) PYBIND11_TLS_GET_VALUE((*internals_pp)->tstate);
        if (!desired_tstate) {
            // nope, we have to create one.
            desired_tstate = PyThreadState_New(si.istate_);
            free_tstate_ = desired_tstate;
#if defined(PYBIND11_DETAILED_ERROR_MESSAGES)
            if (!desired_tstate) {
                pybind11_fail("subinterpreter_scoped_activate: could not create thread state!");
            }
#endif
            PYBIND11_TLS_REPLACE_VALUE((*internals_pp)->tstate, desired_tstate);
        }
    } else {
        desired_tstate = PyThreadState_New(si.istate_);
        free_tstate_ = desired_tstate;
    }

    // make the interpreter active and acquire the GIL
    old_tstate_ = PyThreadState_Swap(desired_tstate);
}

subinterpreter_scoped_activate::~subinterpreter_scoped_activate() {
    if (simple_gil_) {
        // We were on this interpreter already, so just make sure the GIL goes back as it was
        PyGILState_Release(gil_state_);
    } else {
        if (free_tstate_) {
#if defined(PYBIND11_DETAILED_ERROR_MESSAGES)
            if (detail::get_thread_state_unchecked() != free_tstate_) {
                pybind11_fail("~subinterpreter_scoped_activate: thread state must be current!");
            }
#endif
            PYBIND11_TLS_DELETE_VALUE(detail::get_internals().tstate);
            PyThreadState_Clear(free_tstate_);
            PyThreadState_DeleteCurrent();
        }

        // Go back the previous interpreter (if any) and acquire THAT gil
        PyThreadState_Swap(old_tstate_);
    }
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
