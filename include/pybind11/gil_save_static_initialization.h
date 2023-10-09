// Copyright (c) 2023 The pybind Community.

#pragma once

#include "detail/common.h"

#include <cassert>
#include <utility>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

// Use the `gil_save_static_initialization` class below instead of the naive
//
//   static auto imported_obj = py::module_::import("module_name"); // BAD, DO NOT USE!
//
// which has two serious issues:
//
//     1. Py_DECREF() calls potentially after the Python interpreter was finalized already, and
//     2. deadlocks in multi-threaded processes.
//
// This alternative will avoid both problems:
//
//   PYBIND11_CONSTINIT static py::gil_save_static_initialization<py::object> obj_importer;
//   auto imported_obj = obj_importer.get([]() { return py::module_::import("module_name"); });
//
// The `get()` argument is meant to be a callable that makes Python C API calls.
//
// `T` can be any C++ type, it does not have to be a Python type.
//
// Main author of this class: jbms@ (original name: LazyInitializeAtLeastOnceDestroyNever)
template <typename T>
class gil_save_static_initialization {
public:
    // PRECONDITION: The GIL must be held when `get()` is called.
    // It is possible that multiple threads execute `get()` with `initialized_`
    // still being false, and thus proceed to execute `initialize()`. This can
    // happen if `initialize()` releases and reacquires the GIL internally.
    // We accept this, and expect the operation to be both idempotent and cheap.
    template <typename Initialize>
    T &get(Initialize &&initialize) {
        if (!initialized_) {
            assert(PyGILState_Check());
            auto value = initialize();
            if (!initialized_) {
                ::new (value_storage_) T(std::move(value));
                initialized_ = true;
            }
        }
        PYBIND11_WARNING_PUSH
#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ < 5
        // Needed for gcc 4.8.5
        PYBIND11_WARNING_DISABLE_GCC("-Wstrict-aliasing")
#endif
        return *reinterpret_cast<T *>(value_storage_);
        PYBIND11_WARNING_POP
    }

    constexpr gil_save_static_initialization() = default;
    PYBIND11_DTOR_CONSTEXPR ~gil_save_static_initialization() = default;

private:
    alignas(T) char value_storage_[sizeof(T)] = {};
    bool initialized_ = false;
};

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
