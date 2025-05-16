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

    subinterpreter_scoped_activate(subinterpreter_scoped_activate &&) = delete;
    subinterpreter_scoped_activate(subinterpreter_scoped_activate const &) = delete;
    subinterpreter_scoped_activate &operator=(subinterpreter_scoped_activate &) = delete;
    subinterpreter_scoped_activate &operator=(subinterpreter_scoped_activate const &) = delete;

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

    /**
    Because a subinterpreter must be destructed using the original PyThreadState returned when it
    was created. However, because that state has TSS/TLS values (just like any PyThreadState) it
    cannot be trivially moved to a different OS thread. If someone moves the subinterpreter and
    destructs it, it may deadlock in Python cleanup.

    So we try to throw here instead, but if there is an active exception then we have to just leak
    the interpreter.
    */
    ~subinterpreter() noexcept(false) {
        if (tstate_) {
#ifdef PY_HAVE_THREAD_NATIVE_ID
            if (PyThread_get_thread_native_id() != tstate_->native_thread_id) {
                auto cur_tstate = detail::get_thread_state_unchecked();
                if (cur_tstate && cur_tstate->interp == tstate_->interp) {
                    // the destructing subinterpreter was active, release the GIL
                    PyThreadState_Swap(nullptr);
                }

                bool throwable;
#    ifndef __cpp_lib_uncaught_exceptions
                // std::uncaught_exception was removed in C++20
                throwable = !std::uncaught_exception();
#    else
                // std::uncaught_exceptions was added in C++14
                throwable = !(std::uncaught_exceptions() > 0);
#    endif

                if (throwable) {
                    throw std::runtime_error("Cannot destruct a subinterpreter on a different "
                                             "thread from the one that created it!");
                }
                return;
            }
#endif

            // switch into the expiring interpreter
            auto old_tstate = PyThreadState_Swap(tstate_);
            bool switch_back = old_tstate && old_tstate->interp != tstate_->interp;

            // make sure we have the GIL for the interpreter we are ending
            (void) PyGILState_Ensure();

            // Get the internals pointer (without creating it if it doesn't exist).  It's possible
            // for the internals to be created during Py_EndInterpreter() (e.g. if a py::capsule
            // calls `get_internals()` during destruction), so we get the pointer-pointer here and
            // check it after.
            auto *&internals_ptr_ptr = detail::get_internals_pp<detail::internals>();
            auto *&local_internals_ptr_ptr = detail::get_internals_pp<detail::local_internals>();
            {
                dict state_dict = detail::get_python_state_dict();
                internals_ptr_ptr
                    = detail::get_internals_pp_from_capsule_in_state_dict<detail::internals>(
                        state_dict, PYBIND11_INTERNALS_ID);
                local_internals_ptr_ptr
                    = detail::get_internals_pp_from_capsule_in_state_dict<detail::local_internals>(
                        state_dict, detail::get_local_internals_id());
            }

            // End it
            Py_EndInterpreter(tstate_);

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
    }

    /// abandon cleanup of this subinterpreter (leak it). this might be needed during
    /// finalization...
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
#if defined(PYBIND11_DETAILED_ERROR_MESSAGES)
    if (!si.istate_) {
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

inline subinterpreter_scoped_activate::~subinterpreter_scoped_activate() {
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
