/*
    tests/test_thread.cpp -- call pybind11 bound methods in threads

    Copyright (c) 2021 Laramie Leavitt (Google LLC) <lar@google.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <pybind11/cast.h>
#include <pybind11/detail/internals.h>
#include <pybind11/pybind11.h>

#include "pybind11_tests.h"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#if defined(PYBIND11_HAS_STD_BARRIER)
#    include <barrier>
#endif

namespace py = pybind11;

namespace {

struct IntStruct {
    explicit IntStruct(int v) : value(v) {};
    ~IntStruct() { value = -value; }
    IntStruct(const IntStruct &) = default;
    IntStruct &operator=(const IntStruct &) = default;

    int value;
};

struct EmptyStruct {};
EmptyStruct SharedInstance;

#ifdef Py_GIL_DISABLED

// Holds detail::internals::mutex on a native thread that never touches the Python C API, so that
// a Python-level call can be checked for independence from the internals lock. The lock is a
// no-op unless Py_GIL_DISABLED, hence the #ifdef. See test_dispatch_does_not_need_internals_lock.
class internals_lock_holder {
public:
    internals_lock_holder() = default;
    internals_lock_holder(const internals_lock_holder &) = delete;
    internals_lock_holder &operator=(const internals_lock_holder &) = delete;
    ~internals_lock_holder() {
        if (thread.joinable()) {
            request_release_and_join();
        }
    }

    // Returns once the native thread holds the internals mutex. The thread releases it when
    // release_and_join() is called, or after `watchdog` if nobody asks: that keeps the process
    // moving when the call under test blocks on the mutex.
    void start(py::detail::pymutex &internals_mutex, std::chrono::milliseconds watchdog) {
        if (thread.joinable()) {
            throw std::runtime_error("internals_lock_holder is already started");
        }
        held = false;
        release_requested = false;
        released_by_watchdog = false;
        thread = std::thread([this, &internals_mutex, watchdog]() {
            internals_mutex.lock();
            {
                std::lock_guard<std::mutex> lock(m);
                held = true;
            }
            cv.notify_all();
            {
                std::unique_lock<std::mutex> lock(m);
                released_by_watchdog
                    = !cv.wait_for(lock, watchdog, [this]() { return release_requested; });
            }
            internals_mutex.unlock();
        });
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [this]() { return held; });
    }

    // Returns true if the watchdog had already released the mutex.
    bool release_and_join() {
        if (!thread.joinable()) {
            throw std::runtime_error("internals_lock_holder is not started");
        }
        return request_release_and_join();
    }

private:
    bool request_release_and_join() {
        {
            std::lock_guard<std::mutex> lock(m);
            release_requested = true;
        }
        cv.notify_all();
        thread.join();
        return released_by_watchdog;
    }

    std::mutex m;
    std::condition_variable cv;
    bool held = false;
    bool release_requested = false;
    bool released_by_watchdog = false;
    std::thread thread;
};

internals_lock_holder &get_internals_lock_holder() {
    static internals_lock_holder holder;
    return holder;
}

// Deliberately raw PyCFunctions, not pybind11 bindings: these must be callable while the
// internals mutex is held, and whether a pybind11 dispatch can be is exactly what the test checks.
PyObject *start_internals_lock_holder(PyObject *, PyObject *watchdog_seconds_obj) {
    const double watchdog_seconds = PyFloat_AsDouble(watchdog_seconds_obj);
    if (watchdog_seconds == -1.0 && PyErr_Occurred() != nullptr) {
        return nullptr;
    }
    const auto watchdog = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::duration<double>(watchdog_seconds));
    // Looked up while attached; the holder thread itself never uses the Python C API.
    auto &internals_mutex = py::detail::get_internals().mutex;
    std::string error;
    PyThreadState *save = PyEval_SaveThread();
    try {
        get_internals_lock_holder().start(internals_mutex, watchdog);
    } catch (const std::exception &e) {
        error = e.what();
    }
    PyEval_RestoreThread(save);
    if (!error.empty()) {
        PyErr_SetString(PyExc_RuntimeError, error.c_str());
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject *release_internals_lock_holder(PyObject *, PyObject *) {
    bool released_by_watchdog = false;
    std::string error;
    PyThreadState *save = PyEval_SaveThread();
    try {
        released_by_watchdog = get_internals_lock_holder().release_and_join();
    } catch (const std::exception &e) {
        error = e.what();
    }
    PyEval_RestoreThread(save);
    if (!error.empty()) {
        PyErr_SetString(PyExc_RuntimeError, error.c_str());
        return nullptr;
    }
    return PyBool_FromLong(released_by_watchdog ? 1 : 0);
}

// Positive control for the test: a call that is known to take the internals lock.
PyObject *take_internals_lock(PyObject *, PyObject *) {
    try {
        py::detail::with_internals([](py::detail::internals &) {});
    } catch (py::error_already_set &e) {
        e.restore();
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyMethodDef internals_lock_methods[]
    = {{"start_internals_lock_holder", start_internals_lock_holder, METH_O, nullptr},
       {"release_internals_lock_holder", release_internals_lock_holder, METH_NOARGS, nullptr},
       {"take_internals_lock", take_internals_lock, METH_NOARGS, nullptr},
       {nullptr, nullptr, 0, nullptr}};

#endif // Py_GIL_DISABLED

} // namespace

TEST_SUBMODULE(thread, m) {
    py::class_<IntStruct>(m, "IntStruct").def(py::init([](const int i) { return IntStruct(i); }));

    // implicitly_convertible uses loader_life_support when an implicit
    // conversion is required in order to lifetime extend the reference.
    //
    // This test should be run with ASAN for better effectiveness.
    py::implicitly_convertible<int, IntStruct>();

    m.def("test", [](int expected, const IntStruct &in) {
        {
            py::gil_scoped_release release;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        if (in.value != expected) {
            throw std::runtime_error("Value changed!!");
        }
    });

    m.def(
        "test_no_gil",
        [](int expected, const IntStruct &in) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            if (in.value != expected) {
                throw std::runtime_error("Value changed!!");
            }
        },
        py::call_guard<py::gil_scoped_release>());

    py::class_<EmptyStruct>(m, "EmptyStruct")
        .def_readonly_static("SharedInstance", &SharedInstance);

#if defined(PYBIND11_HAS_STD_BARRIER)
    // In the free-threaded build, during PyThreadState_Clear, removing the thread from the biased
    // reference counting table may call destructors. Make sure that it doesn't crash.
    m.def("test_pythread_state_clear_destructor", [](py::type cls) {
        py::handle obj;

        std::barrier barrier{2};
        std::thread thread1{[&]() {
            py::gil_scoped_acquire gil;
            obj = cls().release();
            barrier.arrive_and_wait();
        }};
        std::thread thread2{[&]() {
            py::gil_scoped_acquire gil;
            barrier.arrive_and_wait();
            // ob_ref_shared becomes negative; transition to the queued state
            obj.dec_ref();
        }};

        // jthread is not supported by Apple Clang
        thread1.join();
        thread2.join();
    });
#endif

    m.attr("defined_PYBIND11_HAS_STD_BARRIER") =
#ifdef PYBIND11_HAS_STD_BARRIER
        true;
#else
        false;
#endif
    m.def("acquire_gil", []() { py::gil_scoped_acquire gil_acquired; });

    // The call under test for test_dispatch_does_not_need_internals_lock: the smallest possible
    // bound function, so that the only pybind11 machinery involved is cpp_function::dispatcher().
    m.def("dispatch_noop", []() {});
#ifdef Py_GIL_DISABLED
    if (PyModule_AddFunctions(m.ptr(), internals_lock_methods) != 0) {
        throw py::error_already_set();
    }
#endif

    // NOTE: std::string_view also uses loader_life_support to ensure that
    // the string contents remain alive, but that's a C++ 17 feature.
}
