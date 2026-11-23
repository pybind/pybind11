// Copyright (c) 2023 The pybind Community.

#pragma once

#include "detail/common.h"
#include "detail/internals.h"
#include "gil.h"

#include <cassert>
#include <cstdint>
#include <mutex>
#include <string>

#if defined(Py_GIL_DISABLED) || defined(PYBIND11_HAS_SUBINTERPRETER_SUPPORT)
#    include <atomic>
#endif
#ifdef PYBIND11_HAS_SUBINTERPRETER_SUPPORT
#    include <memory>
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

PYBIND11_NAMESPACE_BEGIN(detail)
#if defined(Py_GIL_DISABLED) || defined(PYBIND11_HAS_SUBINTERPRETER_SUPPORT)
using atomic_bool = std::atomic_bool;
#else
using atomic_bool = bool;
#endif
PYBIND11_NAMESPACE_END(detail)

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

// Prefix for storage keys in the interpreter state dict.
#define PYBIND11_CALL_ONCE_STORAGE_KEY_PREFIX PYBIND11_INTERNALS_ID "_call_once_storage__"

#ifndef PYBIND11_HAS_SUBINTERPRETER_SUPPORT
// Subinterpreter support is disabled.
// In this case, we can store the result globally, because there is only a single interpreter.
//
// The stored result is never destroyed, to avoid destructor calls after the Python interpreter
// was finalized. If the interpreter is finalized and initialized again, the result is stored
// again and the old result is leaked.

PYBIND11_NAMESPACE_BEGIN(detail)

// Lives in the interpreter state dict. The interpreter destroys it during finalization, and
// then the owner must forget its stored result.
template <typename Owner>
struct call_once_reset_on_finalize {
    Owner *owner = nullptr;

    call_once_reset_on_finalize() = default;
    ~call_once_reset_on_finalize() {
        if (owner != nullptr) {
            owner->reset();
        }
    }
    call_once_reset_on_finalize(const call_once_reset_on_finalize &) = delete;
    call_once_reset_on_finalize(call_once_reset_on_finalize &&) = delete;
    call_once_reset_on_finalize &operator=(const call_once_reset_on_finalize &) = delete;
    call_once_reset_on_finalize &operator=(call_once_reset_on_finalize &&) = delete;
};

PYBIND11_NAMESPACE_END(detail)

// The helper must have static storage duration and stay loaded until the interpreter is
// finalized.
template <typename T>
class gil_safe_call_once_and_store {
public:
    // PRECONDITION: The GIL must be held when `call_once_and_store_result()` is called.
    //
    // NOTE: The second parameter (finalize callback) is intentionally unused when subinterpreter
    // support is disabled. In that case, storage is process-global and intentionally leaked to
    // avoid calling destructors after the Python interpreter has been finalized.
    template <typename Callable>
    gil_safe_call_once_and_store &call_once_and_store_result(Callable &&fn,
                                                             void (*)(T &) /*unused*/ = nullptr) {
        if (!is_initialized_) { // This read is guarded by the GIL.
            // Multiple threads may enter here, because the GIL is released in the next line and
            // CPython API calls in the `fn()` call below may release and reacquire the GIL.
            gil_scoped_release gil_rel; // Needed to establish lock ordering.
            std::call_once(once_flag_, [&] {
                // Only one thread will ever enter here.
                gil_scoped_acquire gil_acq;
                // Registered first: if this throws, there is no stored result to roll back.
                detail::atomic_get_or_create_in_state_dict<reset_type>(
                    (PYBIND11_CALL_ONCE_STORAGE_KEY_PREFIX
                     + std::to_string(reinterpret_cast<std::uintptr_t>(this)))
                        .c_str())
                    .first->owner = this;
                ::new (storage_) T(fn()); // fn may release, but will reacquire, the GIL.
                is_initialized_ = true;   // This write is guarded by the GIL.
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
    // The instance is a global static, so its destructor runs when the process
    // is terminating. Therefore, do nothing here because the Python interpreter
    // may have been finalized already.
    PYBIND11_DTOR_CONSTEXPR ~gil_safe_call_once_and_store() = default;

    // Disable copy and move operations.
    gil_safe_call_once_and_store(const gil_safe_call_once_and_store &) = delete;
    gil_safe_call_once_and_store(gil_safe_call_once_and_store &&) = delete;
    gil_safe_call_once_and_store &operator=(const gil_safe_call_once_and_store &) = delete;
    gil_safe_call_once_and_store &operator=(gil_safe_call_once_and_store &&) = delete;

private:
    using reset_type = detail::call_once_reset_on_finalize<gil_safe_call_once_and_store>;
    friend reset_type;

    // Called during interpreter finalization. The stored result belongs to that interpreter.
    // It is leaked, and the next `call_once_and_store_result()` stores a new one.
    void reset() {
        is_initialized_ = false;
        once_flag_.~once_flag();
        ::new (&once_flag_) std::once_flag();
    }

    // The global static storage (per-process) when subinterpreter support is disabled.
    alignas(T) char storage_[sizeof(T)] = {};
    std::once_flag once_flag_;

    // The `is_initialized_`-`storage_` pair is very similar to `std::optional`,
    // but the latter does not have the triviality properties of former,
    // therefore `std::optional` is not a viable alternative here.
    detail::atomic_bool is_initialized_{false};
};
#else
// Subinterpreter support is enabled.
// In this case, we should store the result per-interpreter instead of globally, because each
// subinterpreter has its own separate state. The cached result may not shareable across
// interpreters (e.g., imported modules and their members).

template <typename T>
class gil_safe_call_once_and_store;

PYBIND11_NAMESPACE_BEGIN(detail)

template <typename T>
struct call_once_storage {
    alignas(T) char storage[sizeof(T)] = {};
    std::once_flag once_flag;
    void (*finalize)(T &) = nullptr;
    std::atomic_bool is_initialized{false};
    // The `gil_safe_call_once_and_store` which caches a pointer into `storage`, if any.
    // The interpreter owns this storage. The owner must outlive it: see the requirement on
    // `gil_safe_call_once_and_store` below.
    gil_safe_call_once_and_store<T> *owner = nullptr;

    call_once_storage() = default;
    ~call_once_storage() {
        // The interpreter destroys this storage, therefore the owner must forget its cache.
        if (owner != nullptr) {
            owner->invalidate_cache(reinterpret_cast<T *>(storage));
        }
        if (is_initialized) {
            if (finalize != nullptr) {
                finalize(*reinterpret_cast<T *>(storage));
            } else {
                reinterpret_cast<T *>(storage)->~T();
            }
        }
    }
    call_once_storage(const call_once_storage &) = delete;
    call_once_storage(call_once_storage &&) = delete;
    call_once_storage &operator=(const call_once_storage &) = delete;
    call_once_storage &operator=(call_once_storage &&) = delete;
};

PYBIND11_NAMESPACE_END(detail)

// The helper must have static storage duration and outlive every interpreter that uses it:
// the interpreter writes to the helper during finalization. Do not unload the shared library
// that contains the helper before that.
// The stored result lives as long as its interpreter. An optional `finalize_fn` can clean up
// the result when that interpreter is destroyed.
template <typename T>
class gil_safe_call_once_and_store {
public:
    // PRECONDITION: The GIL must be held when `call_once_and_store_result()` is called.
    template <typename Callable>
    gil_safe_call_once_and_store &call_once_and_store_result(Callable &&fn,
                                                             void (*finalize_fn)(T &) = nullptr) {
        if (get_cached_storage() == nullptr) {
            // Multiple threads may enter here, because the GIL is released in the next line and
            // CPython API calls in the `fn()` call below may release and reacquire the GIL.
            gil_scoped_release gil_rel; // Needed to establish lock ordering.
            // There can be multiple threads going through here.
            storage_type *value = nullptr;
            {
                gil_scoped_acquire gil_acq; // Restore lock ordering.
                // This function is thread-safe under free-threading.
                value = get_or_create_storage_in_state_dict();
            }
            assert(value != nullptr);
            std::call_once(value->once_flag, [&] {
                // Only one thread will ever enter here.
                gil_scoped_acquire gil_acq;
                // fn may release, but will reacquire, the GIL.
                ::new (value->storage) T(fn());
                value->finalize = finalize_fn;
                value->is_initialized = true;
                // Let the storage reset the cache below when the interpreter destroys it. Set
                // only here: `get_stored()` can run concurrently in threads of one interpreter.
                value->owner = this;
                last_storage_ptr_ = reinterpret_cast<T *>(value->storage);
            });
        }
        // Intentionally not returning `T &` to ensure the calling code is self-documenting.
        return *this;
    }

    // This must only be called after `call_once_and_store_result()` was called.
    T &get_stored() {
        if (auto *cached = get_cached_storage()) {
            return *cached;
        }
        // The cache is not published again here. It can thus stay empty if another interpreter
        // replaced the pointer and then cleared it. This is accepted: `get_stored()` is then
        // slower, but correct.
        gil_scoped_acquire gil_acq;
        auto *value = get_or_create_storage_in_state_dict();
        if (!value->is_initialized) {
            pybind11_fail("gil_safe_call_once_and_store::get_stored(): no stored value in this "
                          "interpreter");
        }
        return *reinterpret_cast<T *>(value->storage);
    }

    constexpr gil_safe_call_once_and_store() = default;
    // The instance is a global static, so its destructor runs when the process
    // is terminating. Therefore, do nothing here because the Python interpreter
    // may have been finalized already.
    PYBIND11_DTOR_CONSTEXPR ~gil_safe_call_once_and_store() = default;

    // Disable copy and move operations because the memory address is used as key.
    gil_safe_call_once_and_store(const gil_safe_call_once_and_store &) = delete;
    gil_safe_call_once_and_store(gil_safe_call_once_and_store &&) = delete;
    gil_safe_call_once_and_store &operator=(const gil_safe_call_once_and_store &) = delete;
    gil_safe_call_once_and_store &operator=(gil_safe_call_once_and_store &&) = delete;

private:
    using storage_type = detail::call_once_storage<T>;
    friend storage_type;

    // Another interpreter can publish its pointer during teardown. Compare and clear in one
    // operation so the destructor only removes its own storage from the cache (PR #6162).
    // No targeted race test yet.
    void invalidate_cache(T *storage_ptr) {
        last_storage_ptr_.compare_exchange_strong(storage_ptr, nullptr);
    }

    T *get_cached_storage() const {
        // Load the pointer first. A subinterpreter sets the flag before it can publish its own
        // pointer. If the flag was read first, a thread of the main interpreter could read
        // "false", then load the pointer of the subinterpreter (PR #6162).
        // No targeted race test yet.
        auto *cached = last_storage_ptr_.load();
        return cached != nullptr && !detail::has_seen_non_main_interpreter() ? cached : nullptr;
    }

    // Get the unique key for this storage instance in the interpreter's state dict.
    // The return type should not be `py::str` because PyObject is interpreter-dependent.
    std::string get_storage_key() const {
        // The instance is expected to be global static, so using its address as unique identifier.
        // The typical usage is like:
        //
        //   PYBIND11_CONSTINIT static gil_safe_call_once_and_store<T> storage;
        //
        return PYBIND11_CALL_ONCE_STORAGE_KEY_PREFIX
               + std::to_string(reinterpret_cast<std::uintptr_t>(this));
    }

    // Get or create per-storage capsule in the current interpreter's state dict.
    // The storage is interpreter-dependent and will not be shared across interpreters.
    storage_type *get_or_create_storage_in_state_dict() {
        return detail::atomic_get_or_create_in_state_dict<storage_type>(get_storage_key().c_str())
            .first;
    }

    // Fast cache for the single-interpreter case. A null pointer means the cache is empty.
    // Initialization and teardown can access it concurrently across interpreters.
    std::atomic<T *> last_storage_ptr_{nullptr};
};
#endif

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
