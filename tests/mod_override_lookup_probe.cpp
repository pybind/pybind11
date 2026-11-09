/*
    tests/mod_override_lookup_probe.cpp -- pybind11 built with an instrumented instance-map
    shard unlock, so that a thread can be parked at the point where the wrapper handle found by
    detail::get_object_handle() is no longer protected by the lock. Used by test_thread.py.

    Redefining PyMutex_Unlock changes pybind11's inline functions, which is why this probe is a
    module of its own rather than part of pybind11_tests.

    Copyright (c) 2026 The pybind Community.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/
#include <Python.h>
#include <atomic>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef Py_GIL_DISABLED
namespace {

// Captures the header's real PyMutex_Unlock before the token is redirected below.
void real_PyMutex_Unlock(PyMutex *m) { PyMutex_Unlock(m); }

std::atomic<bool> armed{false};
std::atomic<bool> parked{false};
std::atomic<bool> release_requested{false};
std::atomic<bool> park_timed_out{false};
std::thread::id armed_thread;
const char *shard_mutexes_begin = nullptr;
const char *shard_mutexes_end = nullptr;

// with_instance_map() releases its shard mutex through this call. The release happens as usual.
// Then, if the calling thread armed the probe, it is parked right here: whatever
// get_object_handle() found under the lock is now unprotected and about to be returned.
void probe_PyMutex_Unlock(PyMutex *m) {
    real_PyMutex_Unlock(m);
    const auto *p = reinterpret_cast<const char *>(m);
    if (!armed.load() || std::this_thread::get_id() != armed_thread || p < shard_mutexes_begin
        || p >= shard_mutexes_end || !armed.exchange(false)) {
        return;
    }
    // Detached while parked, so that a stop-the-world (e.g. GC) elsewhere cannot deadlock.
    PyThreadState *tstate = PyEval_SaveThread();
    parked.store(true);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (!release_requested.load()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            park_timed_out.store(true);
            break;
        }
        std::this_thread::yield();
    }
    PyEval_RestoreThread(tstate);
}

} // namespace

#    undef PyMutex_Unlock
#    define PyMutex_Unlock probe_PyMutex_Unlock
#endif

#include <pybind11/pybind11.h>

#ifdef Py_GIL_DISABLED
#    undef PyMutex_Unlock
#endif

namespace py = pybind11;

namespace {

struct Base {
    Base() = default;
    Base(const Base &) = delete;
    Base &operator=(const Base &) = delete;
    virtual ~Base() = default;
    virtual int ping() { return 1; }
};

struct PyBase : Base {
    int ping() override { PYBIND11_OVERRIDE(int, Base, ping, ); }
};

#ifdef Py_GIL_DISABLED

std::shared_ptr<Base> retained; // C++ co-owns the object: it outlives its Python wrapper.
PyObject *wrapper_weakref = nullptr;
std::thread worker;
bool lookup_returned_nonnull = false;
bool wrapper_alive_after_lookup = false;
int ping_result = -1;
std::string worker_error;

bool wrapper_alive() {
    PyObject *strong = nullptr;
    bool alive = PyWeakref_GetRef(wrapper_weakref, &strong) == 1;
    Py_XDECREF(strong);
    return alive;
}

void worker_main(const std::shared_ptr<Base> &owner, bool call_override) {
    py::gil_scoped_acquire gil;
    try {
        const auto *tinfo = py::detail::get_type_info(typeid(Base));
        armed_thread = std::this_thread::get_id();
        armed.store(true);
        if (call_override) {
            // PYBIND11_OVERRIDE -> get_type_override() -> get_object_handle(), after which
            // get_type_override() works with whatever handle it got.
            ping_result = owner->ping();
        } else {
            // The lookup alone. Its result is never dereferenced here; the weakref says whether
            // the wrapper it refers to is still alive.
            auto found = py::detail::get_object_handle(owner.get(), tinfo);
            lookup_returned_nonnull = static_cast<bool>(found);
            wrapper_alive_after_lookup = wrapper_alive();
        }
        armed.store(false);
    } catch (const std::exception &e) {
        worker_error = e.what();
    }
}

#endif // Py_GIL_DISABLED

} // namespace

PYBIND11_MODULE(mod_override_lookup_probe, m, py::mod_gil_not_used()) {
    py::class_<Base, PyBase, std::shared_ptr<Base>>(m, "Base").def(py::init<>());

#ifdef Py_GIL_DISABLED
    m.def("retain", [](std::shared_ptr<Base> obj) { retained = std::move(obj); });

    // Starts the worker; it parks itself once get_object_handle() has released the shard mutex.
    m.def(
        "start",
        [](py::handle weakref, bool call_override) {
            if (worker.joinable() || !retained) {
                throw std::runtime_error("start(): call retain() first, finish() before reuse");
            }
            auto &internals = py::detail::get_internals();
            auto *shards = internals.instance_shards.get();
            shard_mutexes_begin = reinterpret_cast<const char *>(shards);
            shard_mutexes_end
                = reinterpret_cast<const char *>(shards + internals.instance_shards_mask + 1);
            parked.store(false);
            release_requested.store(false);
            park_timed_out.store(false);
            lookup_returned_nonnull = false;
            wrapper_alive_after_lookup = false;
            ping_result = -1;
            worker_error.clear();
            Py_XSETREF(wrapper_weakref, Py_NewRef(weakref.ptr()));
            auto owner = retained;
            worker = std::thread([owner, call_override] { worker_main(owner, call_override); });
        },
        py::arg("weakref"),
        py::arg("call_override"));

    m.def("wait_parked", [] {
        py::gil_scoped_release nogil;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!parked.load() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        return parked.load();
    });

    m.def("release", [] { release_requested.store(true); });

    m.def("finish", [] {
        {
            py::gil_scoped_release nogil;
            if (worker.joinable()) {
                worker.join();
            }
        }
        py::dict result;
        result["lookup_returned_nonnull"] = lookup_returned_nonnull;
        result["wrapper_alive_after_lookup"] = wrapper_alive_after_lookup;
        result["ping_result"] = ping_result;
        result["park_timed_out"] = park_timed_out.load();
        result["worker_error"] = worker_error;
        Py_CLEAR(wrapper_weakref);
        retained.reset();
        return result;
    });
#endif
}
