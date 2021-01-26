// Copyright (c) 2020-2021 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

/* Proof-of-Concept for smart pointer interoperability.

High-level aspects:

* Support all `unique_ptr`, `shared_ptr` interops that are feasible.

* Cleanly and clearly report all interops that are infeasible.

* Meant to fit into a `PyObject`, as a holder for C++ objects.

* Support a system design that makes it impossible to trigger
  C++ Undefined Behavior, especially from Python.

* Support a system design with clean runtime inheritance casting. From this
  it follows that the `smart_holder` needs to be type-erased (`void*`).

* Handling of RTTI for the type-erased held pointer is NOT implemented here.
  It is the responsibility of the caller to ensure that `static_cast<T *>`
  is well-formed when calling `as_*` member functions. Inheritance casting
  needs to be handled in a different layer (similar to the code organization
  in boost/python/object/inheritance.hpp).

Details:

* The "root holder" chosen here is a `shared_ptr<void>` (named `vptr` in this
  implementation). This choice is practically inevitable because `shared_ptr`
  has only very limited support for inspecting and accessing its deleter.

* If created from a raw pointer, or a `unique_ptr` without a custom deleter,
  `vptr` always uses a custom deleter, to support `unique_ptr`-like disowning.
  The custom deleters can be extended to included life-time managment for
  external objects (e.g. `PyObject`).

* If created from an external `shared_ptr`, or a `unique_ptr` with a custom
  deleter, including life-time management for external objects is infeasible.

* The smart_holder is movable but not copyable, as a consequence of using
  unique_ptr for the vptr_deleter_armed_flag_ptr. Note that the bool for
  the flag has to live on the heap, for the smart_holder to be movable.
  unique_ptr is a great fit for this situation.
*/

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>

// pybindit = Python Bindings Innovation Track.
// Currently not in pybind11 namespace to signal that this POC does not depend
// on any existing pybind11 functionality.
namespace pybindit {
namespace memory {

template <typename T>
struct guarded_builtin_delete {
    bool *flag_ptr;
    explicit guarded_builtin_delete(bool *armed_flag_ptr) : flag_ptr{armed_flag_ptr} {}
    void operator()(T *raw_ptr) {
        if (*flag_ptr)
            delete raw_ptr;
    }
};

template <typename T, typename D>
struct guarded_custom_deleter {
    bool *flag_ptr;
    explicit guarded_custom_deleter(bool *armed_flag_ptr) : flag_ptr{armed_flag_ptr} {}
    void operator()(T *raw_ptr) {
        if (*flag_ptr)
            D()(raw_ptr);
    }
};

struct smart_holder {
    const std::type_info *rtti_uqp_del;
    std::unique_ptr<bool> vptr_deleter_armed_flag_ptr;
    std::shared_ptr<void> vptr;
    bool vptr_is_using_noop_deleter : 1;
    bool vptr_is_using_builtin_delete : 1;
    bool vptr_is_external_shared_ptr : 1;
    bool is_populated : 1;

    smart_holder()
        : rtti_uqp_del{nullptr}, vptr_is_using_noop_deleter{false},
          vptr_is_using_builtin_delete{false}, vptr_is_external_shared_ptr{false}, is_populated{
                                                                                       false} {}

    explicit smart_holder(bool vptr_deleter_armed_flag)
        : rtti_uqp_del{nullptr}, vptr_deleter_armed_flag_ptr{new bool{vptr_deleter_armed_flag}},
          vptr_is_using_noop_deleter{false}, vptr_is_using_builtin_delete{false},
          vptr_is_external_shared_ptr{false}, is_populated{false} {}

    bool has_pointee() const { return vptr.get() != nullptr; }

    void ensure_is_populated(const char *context) const {
        if (!is_populated) {
            throw std::runtime_error(std::string("Unpopulated holder (") + context + ").");
        }
    }

    template <typename D>
    void ensure_compatible_rtti_uqp_del(const char *context) const {
        if (!rtti_uqp_del) {
            throw std::runtime_error(std::string("Missing unique_ptr deleter (") + context + ").");
        }
        const std::type_info *rtti_requested = &typeid(D);
        if (!(*rtti_requested == *rtti_uqp_del)) {
            throw std::runtime_error(std::string("Incompatible unique_ptr deleter (") + context
                                     + ").");
        }
    }

    void ensure_has_pointee(const char *context) const {
        if (!has_pointee()) {
            throw std::runtime_error(std::string("Disowned holder (") + context + ").");
        }
    }

    void ensure_vptr_is_using_builtin_delete(const char *context) const {
        if (vptr_is_external_shared_ptr) {
            throw std::runtime_error(std::string("Cannot disown external shared_ptr (") + context
                                     + ").");
        }
        if (vptr_is_using_noop_deleter) {
            throw std::runtime_error(std::string("Cannot disown non-owning holder (") + context
                                     + ").");
        }
        if (!vptr_is_using_builtin_delete) {
            throw std::runtime_error(std::string("Cannot disown custom deleter (") + context
                                     + ").");
        }
    }

    void ensure_use_count_1(const char *context) const {
        if (vptr.get() == nullptr) {
            throw std::runtime_error(std::string("Cannot disown nullptr (") + context + ").");
        }
        // In multithreaded environments accessing use_count can lead to
        // race conditions, but in the context of Python it is a bug (elsewhere)
        // if the Global Interpreter Lock (GIL) is not being held when this code
        // is reached.
        if (vptr.use_count() != 1) {
            throw std::runtime_error(std::string("Cannot disown use_count != 1 (") + context
                                     + ").");
        }
    }

    template <typename T>
    static smart_holder from_raw_ptr_unowned(T *raw_ptr) {
        smart_holder hld(false);
        hld.vptr.reset(raw_ptr, guarded_builtin_delete<T>(hld.vptr_deleter_armed_flag_ptr.get()));
        hld.vptr_is_using_noop_deleter = true;
        hld.is_populated               = true;
        return hld;
    }

    template <typename T>
    T *as_raw_ptr_unowned() const {
        return static_cast<T *>(vptr.get());
    }

    template <typename T>
    T &as_lvalue_ref() const {
        static const char *context = "as_lvalue_ref";
        ensure_is_populated(context);
        ensure_has_pointee(context);
        return *as_raw_ptr_unowned<T>();
    }

    template <typename T>
    T &&as_rvalue_ref() const {
        static const char *context = "as_rvalue_ref";
        ensure_is_populated(context);
        ensure_has_pointee(context);
        return std::move(*as_raw_ptr_unowned<T>());
    }

    template <typename T>
    static smart_holder from_raw_ptr_take_ownership(T *raw_ptr) {
        smart_holder hld(true);
        hld.vptr.reset(raw_ptr, guarded_builtin_delete<T>(hld.vptr_deleter_armed_flag_ptr.get()));
        hld.vptr_is_using_builtin_delete = true;
        hld.is_populated                 = true;
        return hld;
    }

    void ensure_can_release_ownership(const char *context = "ensure_can_release_ownership") {
        ensure_vptr_is_using_builtin_delete(context);
        ensure_use_count_1(context);
    }

    // Caller is responsible for calling ensure_can_release_ownership().
    void release_ownership() {
        *vptr_deleter_armed_flag_ptr = false;
        vptr.reset();
    }

    template <typename T>
    T *as_raw_ptr_release_ownership(const char *context = "as_raw_ptr_release_ownership") {
        ensure_can_release_ownership(context);
        T *raw_ptr = as_raw_ptr_unowned<T>();
        release_ownership();
        return raw_ptr;
    }

    template <typename T>
    static smart_holder from_unique_ptr(std::unique_ptr<T> &&unq_ptr) {
        smart_holder hld(true);
        hld.vptr.reset(unq_ptr.get(),
                       guarded_builtin_delete<T>(hld.vptr_deleter_armed_flag_ptr.get()));
        unq_ptr.release();
        hld.vptr_is_using_builtin_delete = true;
        hld.is_populated                 = true;
        return hld;
    }

    template <typename T>
    std::unique_ptr<T> as_unique_ptr() {
        return std::unique_ptr<T>(as_raw_ptr_release_ownership<T>("as_unique_ptr"));
    }

    template <typename T, typename D>
    static smart_holder from_unique_ptr_with_deleter(std::unique_ptr<T, D> &&unq_ptr) {
        smart_holder hld(true);
        hld.rtti_uqp_del = &typeid(D);
        hld.vptr.reset(unq_ptr.get(),
                       guarded_custom_deleter<T, D>(hld.vptr_deleter_armed_flag_ptr.get()));
        unq_ptr.release();
        hld.is_populated = true;
        return hld;
    }

    template <typename T, typename D>
    std::unique_ptr<T, D> as_unique_ptr_with_deleter() {
        static const char *context = "as_unique_ptr_with_deleter";
        ensure_compatible_rtti_uqp_del<D>(context);
        ensure_use_count_1(context);
        T *raw_ptr                   = as_raw_ptr_unowned<T>();
        *vptr_deleter_armed_flag_ptr = false;
        vptr.reset();
        return std::unique_ptr<T, D>(raw_ptr);
    }

    template <typename T>
    static smart_holder from_shared_ptr(std::shared_ptr<T> shd_ptr) {
        smart_holder hld;
        hld.vptr                        = std::static_pointer_cast<void>(shd_ptr);
        hld.vptr_is_external_shared_ptr = true;
        hld.is_populated                = true;
        return hld;
    }

    template <typename T>
    std::shared_ptr<T> as_shared_ptr() const {
        return std::static_pointer_cast<T>(vptr);
    }
};

} // namespace memory
} // namespace pybindit
