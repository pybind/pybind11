// Copyright (c) 2023 The pybind Community.

#pragma once

#include "detail/common.h"
#include "gil.h"

#include <atomic>
#include <cassert>
#include <mutex>
#include <utility>

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
// The `call_once_and_store_result()` argument is meant to be a callable that
// makes Python C API calls.
//
// `T` can be any C++ type, it does not have to be a Python type.
template <typename T>
class gil_safe_call_once_and_store {
public:
    // PRECONDITION: The GIL must be held when `get_stored()` is called.
    template <typename Callable>
    gil_safe_call_once_and_store &call_once_and_store_result(Callable &&fn) {
        if (!is_initialized_.load(std::memory_order_acquire)) {
            gil_scoped_release gil_rel;
            std::call_once(once_flag_, [&] {
                gil_scoped_acquire gil_acq;
                ::new (storage_) T(fn());
                is_initialized_.store(true, std::memory_order_release);
            });
        }
        return *this;
    }

    // This must only be called after `call_once_and_store()` was called.
    T &get_stored() const {
        assert(is_initialized_.load(std::memory_order_relaxed));
        PYBIND11_WARNING_PUSH
#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ < 5
        // Needed for gcc 4.8.5
        PYBIND11_WARNING_DISABLE_GCC("-Wstrict-aliasing")
#endif
        return *reinterpret_cast<T *>(storage_);
        PYBIND11_WARNING_POP
    }

    constexpr gil_safe_call_once_and_store() = default;
    PYBIND11_DTOR_CONSTEXPR ~gil_safe_call_once_and_store() = default;

private:
    alignas(T) mutable char storage_[sizeof(T)] = {};
    std::once_flag once_flag_ = {};
    std::atomic<bool> is_initialized_ = {};
};

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
