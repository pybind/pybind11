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
  The custom deleters could be extended to included life-time managment for
  external objects (e.g. `PyObject`).

* If created from an external `shared_ptr`, or a `unique_ptr` with a custom
  deleter, including life-time management for external objects is infeasible.

* By choice, the smart_holder is movable but not copyable, to keep the design
  simple, and to guard against accidental copying overhead.
*/

#pragma once

#include <cassert>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>

//#include <iostream>
//inline void to_cout(std::string msg) { std::cout << msg << std::endl; }
inline void to_cout(std::string) { }

// pybindit = Python Bindings Innovation Track.
// Currently not in pybind11 namespace to signal that this POC does not depend
// on any existing pybind11 functionality.
namespace pybindit {
namespace memory {

inline int shared_from_this_status(...) { return 0; }

template <typename AnyBaseOfT>
#if defined(__cpp_lib_enable_shared_from_this) && (!defined(_MSC_VER) || _MSC_VER >= 1912)
inline int shared_from_this_status(const std::enable_shared_from_this<AnyBaseOfT> *ptr) {
  if (ptr->weak_from_this().lock()) return 1;
  return -1;
}
#else
inline int shared_from_this_status(const std::enable_shared_from_this<AnyBaseOfT> *) {
  return 999;
}
#endif

struct smart_holder;

struct guarded_delete {
    smart_holder *hld            = nullptr;
    void (*callback_ptr)(void *) = nullptr;
    void *callback_arg           = nullptr;
    void (*shd_ptr_reset_fptr)(std::shared_ptr<void>&, void *, guarded_delete &&);
    void (*del_fptr)(void *);
    bool armed_flag;
    guarded_delete(void (*shd_ptr_reset_fptr)(std::shared_ptr<void>&, void *, guarded_delete &&), void (*del_fptr)(void *), bool armed_flag)
        : shd_ptr_reset_fptr{shd_ptr_reset_fptr}, del_fptr{del_fptr}, armed_flag{armed_flag} {
to_cout("LOOOK guarded_delete ctor " + std::to_string(__LINE__) + " " + __FILE__);
        }
    void operator()(void *raw_ptr) const;
};

template <typename T>
inline void shd_ptr_reset(std::shared_ptr<void>& shd_ptr, void *raw_ptr, guarded_delete &&gdel) {
    shd_ptr.reset(static_cast<T *>(raw_ptr), gdel);
}

template <typename T, typename std::enable_if<std::is_destructible<T>::value, int>::type = 0>
inline void builtin_delete_if_destructible(void *raw_ptr) {
    delete static_cast<T *>(raw_ptr);
}

template <typename T, typename std::enable_if<!std::is_destructible<T>::value, int>::type = 0>
inline void builtin_delete_if_destructible(void *) {
    // This noop operator is needed to avoid a compilation error (for `delete raw_ptr;`), but
    // throwing an exception from a destructor will std::terminate the process. Therefore the
    // runtime check for lifetime-management correctness is implemented elsewhere (in
    // ensure_pointee_is_destructible()).
}

template <typename T>
guarded_delete make_guarded_builtin_delete(bool armed_flag) {
    return guarded_delete(shd_ptr_reset<T>, builtin_delete_if_destructible<T>, armed_flag);
}

template <typename T, typename D>
inline void custom_delete(void *raw_ptr) {
    D()(static_cast<T *>(raw_ptr));
}

template <typename T, typename D>
guarded_delete make_guarded_custom_deleter(bool armed_flag) {
    return guarded_delete(shd_ptr_reset<T>, custom_delete<T, D>, armed_flag);
};

// Trick to keep the smart_holder memory footprint small.
struct noop_deleter_acting_as_weak_ptr_owner {
    std::weak_ptr<void> passenger;
    void operator()(void *) {};
};

template <typename T>
inline bool is_std_default_delete(const std::type_info &rtti_deleter) {
    return rtti_deleter == typeid(std::default_delete<T>)
           || rtti_deleter == typeid(std::default_delete<T const>);
}

struct smart_holder {
    const std::type_info *rtti_uqp_del = nullptr;
    std::shared_ptr<void> vptr;
    bool vptr_is_using_noop_deleter : 1;
    bool vptr_is_using_builtin_delete : 1;
    bool vptr_is_external_shared_ptr : 1;
    bool is_populated : 1;
    bool is_disowned : 1;
    bool vptr_is_released : 1;
    bool pointee_depends_on_holder_owner : 1; // SMART_HOLDER_WIP: See PR #2839.

    // Design choice: smart_holder is movable but not copyable.
    smart_holder(smart_holder &&)      = default;
    smart_holder(const smart_holder &) = delete;
    smart_holder &operator=(smart_holder &&) = delete;
    smart_holder &operator=(const smart_holder &) = delete;

    smart_holder()
        : vptr_is_using_noop_deleter{false}, vptr_is_using_builtin_delete{false},
          vptr_is_external_shared_ptr{false}, is_populated{false}, is_disowned{false},
          vptr_is_released{false}, pointee_depends_on_holder_owner{false} {}

    bool has_pointee() const { return vptr != nullptr; }

    template <typename T>
    static void ensure_pointee_is_destructible(const char *context) {
        if (!std::is_destructible<T>::value)
            throw std::invalid_argument(std::string("Pointee is not destructible (") + context
                                        + ").");
    }

    void ensure_is_populated(const char *context) const {
        if (!is_populated) {
            throw std::runtime_error(std::string("Unpopulated holder (") + context + ").");
        }
    }
    void ensure_is_not_disowned(const char *context) const {
        if (is_disowned) {
            throw std::runtime_error(std::string("Holder was disowned already (") + context
                                     + ").");
        }
    }

    void ensure_vptr_is_using_builtin_delete(const char *context) const {
        if (vptr_is_external_shared_ptr) {
            throw std::invalid_argument(std::string("Cannot disown external shared_ptr (")
                                        + context + ").");
        }
        if (vptr_is_using_noop_deleter) {
            throw std::invalid_argument(std::string("Cannot disown non-owning holder (") + context
                                        + ").");
        }
        if (!vptr_is_using_builtin_delete) {
            throw std::invalid_argument(std::string("Cannot disown custom deleter (") + context
                                        + ").");
        }
    }

    template <typename T, typename D>
    void ensure_compatible_rtti_uqp_del(const char *context) const {
        const std::type_info *rtti_requested = &typeid(D);
        if (!rtti_uqp_del) {
            if (!is_std_default_delete<T>(*rtti_requested)) {
                throw std::invalid_argument(std::string("Missing unique_ptr deleter (") + context
                                            + ").");
            }
            ensure_vptr_is_using_builtin_delete(context);
        } else if (!(*rtti_requested == *rtti_uqp_del)) {
            throw std::invalid_argument(std::string("Incompatible unique_ptr deleter (") + context
                                        + ").");
        }
    }

    void ensure_has_pointee(const char *context) const {
        if (!has_pointee()) {
            throw std::invalid_argument(std::string("Disowned holder (") + context + ").");
        }
    }

    void ensure_use_count_1(const char *context) const {
        if (vptr == nullptr) {
            throw std::invalid_argument(std::string("Cannot disown nullptr (") + context + ").");
        }
        // In multithreaded environments accessing use_count can lead to
        // race conditions, but in the context of Python it is a bug (elsewhere)
        // if the Global Interpreter Lock (GIL) is not being held when this code
        // is reached.
        // SMART_HOLDER_WIP: IMPROVABLE: assert(GIL is held).
        if (vptr.use_count() != 1) {
            throw std::invalid_argument(std::string("Cannot disown use_count != 1 (") + context
                                        + ").");
        }
    }

    void reset_vptr_deleter_armed_flag(bool armed_flag) const {
to_cout("LOOOK smart_holder reset_vptr_deleter_armed_flag " + std::to_string(__LINE__) + " " + __FILE__);
        auto vptr_del_fptr = std::get_deleter<guarded_delete>(vptr);
        if (vptr_del_fptr == nullptr) {
            throw std::runtime_error(
                "smart_holder::reset_vptr_deleter_armed_flag() called in an invalid context.");
        }
        vptr_del_fptr->armed_flag = armed_flag;
    }

    static smart_holder from_raw_ptr_unowned(void *raw_ptr) {
to_cout("LOOOK smart_holder from_raw_ptr_unowned " + std::to_string(__LINE__) + " " + __FILE__);
        smart_holder hld;
        hld.vptr.reset(raw_ptr, [](void *) {});
        hld.vptr_is_using_noop_deleter = true;
        hld.is_populated               = true;
        return hld;
    }

    template <typename T>
    T *as_raw_ptr_unowned() const {
to_cout("LOOOK smart_holder as_raw_ptr_unowned " + std::to_string(__LINE__) + " " + __FILE__);
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
to_cout("");
to_cout("LOOOK smart_holder from_raw_ptr_take_ownership " + std::to_string(__LINE__) + " " + __FILE__);
        ensure_pointee_is_destructible<T>("from_raw_ptr_take_ownership");
        smart_holder hld;
        hld.vptr.reset(static_cast<void *>(raw_ptr), make_guarded_builtin_delete<T>(true));
        hld.vptr_is_using_builtin_delete = true;
        hld.is_populated                 = true;
        return hld;
    }

    // Caller is responsible for ensuring preconditions (SMART_HOLDER_WIP: details).
    void disown() {
        reset_vptr_deleter_armed_flag(false);
        is_disowned = true;
    }

    // Caller is responsible for ensuring preconditions (SMART_HOLDER_WIP: details).
    void reclaim_disowned() {
        reset_vptr_deleter_armed_flag(true);
        is_disowned = false;
    }

    // Caller is responsible for ensuring preconditions (SMART_HOLDER_WIP: details).
    void release_disowned() { vptr.reset(); }

    // SMART_HOLDER_WIP: review this function.
    void ensure_can_release_ownership(const char *context = "ensure_can_release_ownership") const {
        ensure_is_not_disowned(context);
        ensure_vptr_is_using_builtin_delete(context);
        ensure_use_count_1(context);
    }

    // Caller is responsible for ensuring preconditions (SMART_HOLDER_WIP: details).
    void release_ownership() {
        reset_vptr_deleter_armed_flag(false);
        release_disowned();
    }

    template <typename T>
    T *as_raw_ptr_release_ownership(const char *context = "as_raw_ptr_release_ownership") {
to_cout("LOOOK smart_holder as_raw_ptr_release_ownership " + std::to_string(__LINE__) + " " + __FILE__);
        ensure_can_release_ownership(context);
        T *raw_ptr = as_raw_ptr_unowned<T>();
        release_ownership();
        return raw_ptr;
    }

    template <typename T, typename D>
    static smart_holder from_unique_ptr(std::unique_ptr<T, D> &&unq_ptr) {
to_cout("LOOOK smart_holder from_unique_ptr " + std::to_string(__LINE__) + " " + __FILE__);
        smart_holder hld;
        hld.rtti_uqp_del                 = &typeid(D);
        hld.vptr_is_using_builtin_delete = is_std_default_delete<T>(*hld.rtti_uqp_del);
        if (hld.vptr_is_using_builtin_delete) {
            hld.vptr.reset(static_cast<void *>(unq_ptr.get()),
                           make_guarded_builtin_delete<T>(true));
        } else {
            hld.vptr.reset(static_cast<void *>(unq_ptr.get()),
                           make_guarded_custom_deleter<T, D>(true));
        }
        unq_ptr.release();
        hld.is_populated = true;
        return hld;
    }

    template <typename T, typename D = std::default_delete<T>>
    std::unique_ptr<T, D> as_unique_ptr() {
to_cout("LOOOK smart_holder as_unique_ptr " + std::to_string(__LINE__) + " " + __FILE__);
        static const char *context = "as_unique_ptr";
        ensure_compatible_rtti_uqp_del<T, D>(context);
        ensure_use_count_1(context);
        T *raw_ptr = as_raw_ptr_unowned<T>();
        release_ownership();
        return std::unique_ptr<T, D>(raw_ptr);
    }

    template <typename T>
    static smart_holder from_shared_ptr(std::shared_ptr<T> shd_ptr) {
to_cout("LOOOK smart_holder from_shared_ptr " + std::to_string(__LINE__) + " " + __FILE__);
        smart_holder hld;
        hld.vptr                        = std::static_pointer_cast<void>(shd_ptr);
        hld.vptr_is_external_shared_ptr = true;
        hld.is_populated                = true;
        return hld;
    }

    template <typename T>
    std::shared_ptr<T> as_shared_ptr() const {
to_cout("LOOOK smart_holder as_shared_ptr " + std::to_string(__LINE__) + " " + __FILE__);
        return std::static_pointer_cast<T>(vptr);
    }
};

inline void guarded_delete::operator()(void *raw_ptr) const {
    if (hld) {
to_cout("RECLAIM guarded_delete call hld " + std::to_string(__LINE__) + " " + __FILE__);
        assert(armed_flag);
        assert(hld->vptr.get() == raw_ptr);
        assert(hld->vptr_is_released);
        (*shd_ptr_reset_fptr)(hld->vptr, hld->vptr.get(), guarded_delete{shd_ptr_reset_fptr, del_fptr, true});
        hld->vptr_is_released = false;
        (*callback_ptr)(callback_arg); // Py_DECREF.
    } else if (armed_flag) {
to_cout("LOOOK guarded_delete call del " + std::to_string(__LINE__) + " " + __FILE__);
        (*del_fptr)(raw_ptr);
    } else {
to_cout("LOOOK guarded_delete call noop " + std::to_string(__LINE__) + " " + __FILE__);
    }
}

} // namespace memory
} // namespace pybindit
