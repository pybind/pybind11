// Copyright (c) 2023 The pybind Community.

#pragma once

#include "detail/common.h"
#include "detail/internals.h"
#include "gil.h"

#include <cassert>
#include <mutex>

#if defined(Py_GIL_DISABLED) || defined(PYBIND11_HAS_SUBINTERPRETER_SUPPORT)
#    include <atomic>

using atomic_bool = std::atomic_bool;
#else
using atomic_bool = bool;
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

// Use the `gil_safe_call_once_and_store` class below instead of the naive
//
//   static auto imported_obj = py::module_::import("module_name"); // BAD, DO NOT USE!
//
// which has two serious issues:
//
//     1. Py_DECREF() calls potentially after the Python interpreter was finalized already, and
//     2. deadlocks in multi-threaded processes (because of missing lock ordering).
//
// The following alternative avoids both problems:
//
//   PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object> storage;
//   auto &imported_obj = storage // Do NOT make this `static`!
//       .call_once_and_store_result([]() {
//           return py::module_::import("module_name");
//       })
//       .get_stored();
//
// The parameter of `call_once_and_store_result()` must be callable. It can make
// CPython API calls, and in particular, it can temporarily release the GIL.
//
// `T` can be any C++ type, it does not have to involve CPython API types.
//
// The behavior with regard to signals, e.g. `SIGINT` (`KeyboardInterrupt`),
// is not ideal. If the main thread is the one to actually run the `Callable`,
// then a `KeyboardInterrupt` will interrupt it if it is running normal Python
// code. The situation is different if a non-main thread runs the
// `Callable`, and then the main thread starts waiting for it to complete:
// a `KeyboardInterrupt` will not interrupt the non-main thread, but it will
// get processed only when it is the main thread's turn again and it is running
// normal Python code. However, this will be unnoticeable for quick call-once
// functions, which is usually the case.
//
// For in-depth background, see docs/advanced/deadlock.md
#ifndef PYBIND11_HAS_SUBINTERPRETER_SUPPORT
template <typename T>
class gil_safe_call_once_and_store {
public:
    // PRECONDITION: The GIL must be held when `call_once_and_store_result()` is called.
    template <typename Callable>
    gil_safe_call_once_and_store &call_once_and_store_result(Callable &&fn,
                                                             void (*finalize_fn)(T &) = nullptr) {

        if (!is_initialized_) { // This read is guarded by the GIL.
            // Multiple threads may enter here, because the GIL is released in the next line and
            // CPython API calls in the `fn()` call below may release and reacquire the GIL.
            gil_scoped_release gil_rel; // Needed to establish lock ordering.
            std::call_once(once_flag_, [&] {
                // Only one thread will ever enter here.
                gil_scoped_acquire gil_acq;
                ::new (storage_) T(fn());   // fn may release, but will reacquire, the GIL.
                finalize_fn_ = finalize_fn; // Store the finalizer.
                is_initialized_ = true;     // This write is guarded by the GIL.
            });
            // All threads will observe `is_initialized_` as true here.
        }
        // Intentionally not returning `T &` to ensure the calling code is self-documenting.
        return *this;
    }

    // This must only be called after `call_once_and_store_result()` was called.
    T &get_stored() {
        assert(is_initialized_);
        PYBIND11_WARNING_PUSH
#    if !defined(__clang__) && defined(__GNUC__) && __GNUC__ < 5
        // Needed for gcc 4.8.5
        PYBIND11_WARNING_DISABLE_GCC("-Wstrict-aliasing")
#    endif
        return *reinterpret_cast<T *>(storage_);
        PYBIND11_WARNING_POP
    }

    constexpr gil_safe_call_once_and_store() = default;
    PYBIND11_DTOR_CONSTEXPR ~gil_safe_call_once_and_store() {
        if (is_initialized_ && finalize_fn_ != nullptr) {
            finalize_fn_(*reinterpret_cast<T *>(storage_));
        }
    }

private:
    // Global static storage (per process) when subinterpreter support is disabled.
    alignas(T) char storage_[sizeof(T)] = {};
    std::once_flag once_flag_;
    void (*finalize_fn_)(T &) = nullptr;

    // The `is_initialized_`-`storage_` pair is very similar to `std::optional`,
    // but the latter does not have the triviality properties of former,
    // therefore `std::optional` is not a viable alternative here.
    atomic_bool is_initialized_{false};
};
#else
// Subinterpreter support is enabled.
// In this case, we should store the result per-interpreter instead of globally, because
// each subinterpreter has its own separate state. The cached object may not shareable
// across interpreters (e.g., imported modules and their members).
template <typename T>
class gil_safe_call_once_and_store {
public:
    // PRECONDITION: The GIL must be held when `call_once_and_store_result()` is called.
    template <typename Callable>
    gil_safe_call_once_and_store &call_once_and_store_result(Callable &&fn,
                                                             void (*finalize_fn)(T &) = nullptr) {
        if (!is_initialized_by_atleast_one_interpreter_
            || detail::get_num_interpreters_seen() > 1) {
            detail::with_internals([&](detail::internals &internals) {
                const void *key = reinterpret_cast<const void *>(this);
                auto &storage_map = internals.call_once_storage_map;
                auto it = storage_map.find(key);
                if (it == storage_map.end()) {
                    gil_scoped_release gil_rel; // Needed to establish lock ordering.
                    {
                        // Only one thread will ever enter here.
                        gil_scoped_acquire gil_acq;
                        auto s = new detail::call_once_storage<T>{};
                        ::new (s->storage) T(fn()); // fn may release, but will reacquire, the GIL.
                        s->finalize = finalize_fn;
                        last_storage_ = reinterpret_cast<T *>(s->storage);
                        storage_map.emplace(key, s);
                    };
                }
                is_initialized_by_atleast_one_interpreter_ = true;
            });
            // All threads will observe `is_initialized_by_atleast_one_interp_` as true here.
        }
        // Intentionally not returning `T &` to ensure the calling code is self-documenting.
        return *this;
    }

    // This must only be called after `call_once_and_store_result()` was called.
    T &get_stored() {
        T *result = last_storage_;
        if (!is_initialized_by_atleast_one_interpreter_
            || detail::get_num_interpreters_seen() > 1) {
            detail::with_internals([&](detail::internals &internals) {
                const void *key = reinterpret_cast<const void *>(this);
                auto &storage_map = internals.call_once_storage_map;
                auto it = storage_map.find(key);
                assert(it != storage_map.end());
                auto *s = static_cast<detail::call_once_storage<T> *>(it->second);
                result = last_storage_ = reinterpret_cast<T *>(s->storage);
            });
        }
        assert(result != nullptr);
        return *result;
    }

    constexpr gil_safe_call_once_and_store() = default;
    PYBIND11_DTOR_CONSTEXPR ~gil_safe_call_once_and_store() {
        if (is_initialized_by_atleast_one_interpreter_) {
            detail::with_internals([&](detail::internals &internals) {
                const void *key = reinterpret_cast<const void *>(this);
                auto &storage_map = internals.call_once_storage_map;
                auto it = storage_map.find(key);
                if (it != storage_map.end()) {
                    delete it->second;
                    storage_map.erase(it);
                }
            });
        }
    }

private:
    // No storage needed when subinterpreter support is enabled.
    // The actual storage is stored in the per-interpreter state dict in
    // `internals.call_once_storage_map`.

    // Fast local cache to avoid repeated lookups when there are no multiple interpreters.
    // This is only valid if there is a single interpreter. Otherwise, it is not used.
    T *last_storage_ = nullptr;
    // This flag is true if the value has been initialized by any interpreter (may not be the
    // current one).
    atomic_bool is_initialized_by_atleast_one_interpreter_{false};
};
#endif
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
