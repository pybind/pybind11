/*
    pybind11/gil.h: GIL implementations

    Copyright (c) 2018 Kitware Inc. <kyle.edwards@kitware.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "detail/common.h"
#include "detail/gil_internals.h"

NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

class basic_gil_impl {
public:
    class acquire {
    public:
        acquire() {
            state = PyGILState_Ensure();
        }

        ~acquire() {
            PyGILState_Release(state);
        }

    private:
        PyGILState_STATE state;
    };

    class release {
    public:
        release() {
            state = PyEval_SaveThread();
        }

        ~release() {
            PyEval_RestoreThread(state);
        }

    private:
        PyThreadState *state;
    };

    /*static bool thread_has_impl() {
        return !!PyGILState_GetThisThreadState();
    }*/
};

class advanced_gil_impl {
public:
    class acquire {
    public:
        PYBIND11_NOINLINE acquire() {
            auto const &internals = detail::get_internals();
            tstate = (PyThreadState *) PyThread_get_key_value(internals.tstate);

            if (!tstate) {
                tstate = PyThreadState_New(internals.istate);
                #if !defined(NDEBUG)
                    if (!tstate)
                        pybind11_fail("scoped_acquire: could not create thread state!");
                #endif
                tstate->gilstate_counter = 0;
                #if PY_MAJOR_VERSION < 3
                    PyThread_delete_key_value(internals.tstate);
                #endif
                PyThread_set_key_value(internals.tstate, tstate);
            } else {
                release = detail::get_thread_state_unchecked() != tstate;
            }

            if (release) {
                /* Work around an annoying assertion in PyThreadState_Swap */
                #if defined(Py_DEBUG)
                    PyInterpreterState *interp = tstate->interp;
                    tstate->interp = nullptr;
                #endif
                PyEval_AcquireThread(tstate);
                #if defined(Py_DEBUG)
                    tstate->interp = interp;
                #endif
            }

            inc_ref();
        }

        void inc_ref() {
            ++tstate->gilstate_counter;
        }

        PYBIND11_NOINLINE void dec_ref() {
            --tstate->gilstate_counter;
            #if !defined(NDEBUG)
                if (detail::get_thread_state_unchecked() != tstate)
                    pybind11_fail("scoped_acquire::dec_ref(): thread state must be current!");
                if (tstate->gilstate_counter < 0)
                    pybind11_fail("scoped_acquire::dec_ref(): reference count underflow!");
            #endif
            if (tstate->gilstate_counter == 0) {
                #if !defined(NDEBUG)
                    if (!release)
                        pybind11_fail("scoped_acquire::dec_ref(): internal error!");
                #endif
                PyThreadState_Clear(tstate);
                PyThreadState_DeleteCurrent();
                PyThread_delete_key_value(detail::get_internals().tstate);
                release = false;
            }
        }

        ~acquire() {
            dec_ref();
            if (release)
               PyEval_SaveThread();
        }

    private:
        PyThreadState *tstate = nullptr;
        bool release = true;
    };

    class release {
    public:
        explicit release(bool disassoc = false) : disassoc(disassoc) {
            // `get_internals()` must be called here unconditionally in order to initialize
            // `internals.tstate` for subsequent `gil_scoped_acquire` calls. Otherwise, an
            // initialization race could occur as multiple threads try `gil_scoped_acquire`.
            const auto &internals = detail::get_internals();
            tstate = PyEval_SaveThread();
            if (disassoc) {
                auto key = internals.tstate;
                #if PY_MAJOR_VERSION < 3
                    PyThread_delete_key_value(key);
                #else
                    PyThread_set_key_value(key, nullptr);
                #endif
            }
        }

        ~release() {
            if (!tstate)
                return;
            PyEval_RestoreThread(tstate);
            if (disassoc) {
                auto key = detail::get_internals().tstate;
                #if PY_MAJOR_VERSION < 3
                    PyThread_delete_key_value(key);
                #endif
                PyThread_set_key_value(key, tstate);
            }
        }

    private:
        PyThreadState *tstate;
        bool disassoc;
    };
};

template<typename Impl>
void select_gil_impl() {
    detail::select_gil_impl<Impl>();
}

NAMESPACE_END(PYBIND11_NAMESPACE)
