/* Proof-of-Concept for smart pointer interoperability.

High-level aspects:

* Support all `unique_ptr`, `shared_ptr` interops that are feasible.

* Cleanly and clearly report all interops that are infeasible.

* Meant to fit into a `PyObject`, as a holder for C++ objects.

* Support a system design that makes it impossible to trigger
  C++ Undefined Behavior, especially from Python.

* Support a system design with clean runtime inheritance casting. From this
  it follows that the `smart_holder` needs to be type-erased (`void*`, RTTI).

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

* The `typename T` between `from` and `as` calls must match exactly.
  Inheritance casting needs to be handled in a different layer (similar
  to the code organization in boost/python/object/inheritance.hpp).
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
  bool* flag_ptr;
  explicit guarded_builtin_delete(bool* guard_flag_ptr)
      : flag_ptr{guard_flag_ptr} {}
  void operator()(T* raw_ptr) {
    if (*flag_ptr) delete raw_ptr;
  }
};

template <typename T, typename D>
struct guarded_custom_deleter {
  bool* flag_ptr;
  explicit guarded_custom_deleter(bool* guard_flag_ptr)
      : flag_ptr{guard_flag_ptr} {}
  void operator()(T* raw_ptr) {
    if (*flag_ptr) D()(raw_ptr);
  }
};

struct smart_holder {
  const std::type_info* rtti_held;
  const std::type_info* rtti_uqp_del;
  std::shared_ptr<void> vptr;
  bool vptr_deleter_guard_flag;
  bool vptr_is_using_noop_deleter : 1;
  bool vptr_is_using_builtin_delete : 1;
  bool vptr_is_external_shared_ptr : 1;

  smart_holder()
      : rtti_held{nullptr},
        rtti_uqp_del{nullptr},
        vptr_deleter_guard_flag{false},
        vptr_is_using_noop_deleter{false},
        vptr_is_using_builtin_delete{false},
        vptr_is_external_shared_ptr{false} {}

  bool has_pointee() const { return vptr.get() != nullptr; }

  template <typename T>
  void ensure_compatible_rtti_held(const char* context) const {
    if (!rtti_held) {
      throw std::runtime_error(std::string("Unpopulated holder (") + context +
                               ").");
    }
    const std::type_info* rtti_requested = &typeid(T);
    if (!(*rtti_requested == *rtti_held)) {
      throw std::runtime_error(std::string("Incompatible type (") + context +
                               ").");
    }
  }

  template <typename D>
  void ensure_compatible_rtti_uqp_del(const char* context) const {
    if (!rtti_uqp_del) {
      throw std::runtime_error(std::string("Missing unique_ptr deleter (") +
                               context + ").");
    }
    const std::type_info* rtti_requested = &typeid(D);
    if (!(*rtti_requested == *rtti_uqp_del)) {
      throw std::runtime_error(
          std::string("Incompatible unique_ptr deleter (") + context + ").");
    }
  }

  void ensure_has_pointee(const char* context) const {
    if (!has_pointee()) {
      throw std::runtime_error(std::string("Disowned holder (") + context +
                               ").");
    }
  }

  void ensure_vptr_is_using_builtin_delete(const char* context) const {
    if (vptr_is_external_shared_ptr) {
      throw std::runtime_error(
          std::string("Cannot disown external shared_ptr (") + context + ").");
    }
    if (vptr_is_using_noop_deleter) {
      throw std::runtime_error(
          std::string("Cannot disown non-owning holder (") + context + ").");
    }
    if (!vptr_is_using_builtin_delete) {
      throw std::runtime_error(std::string("Cannot disown custom deleter (") +
                               context + ").");
    }
  }

  void ensure_use_count_1(const char* context) const {
    if (vptr.get() == nullptr) {
      throw std::runtime_error(std::string("Cannot disown nullptr (") +
                               context + ").");
    }
    if (vptr.use_count() != 1) {
      throw std::runtime_error(std::string("Cannot disown use_count != 1 (") +
                               context + ").");
    }
  }

  template <typename T>
  static smart_holder from_raw_ptr_unowned(T* raw_ptr) {
    smart_holder hld;
    hld.rtti_held = &typeid(T);
    hld.vptr_is_using_noop_deleter = true;
    hld.vptr.reset(raw_ptr,
                   guarded_builtin_delete<T>(&hld.vptr_deleter_guard_flag));
    return hld;
  }

  template <typename T>
  T* as_raw_ptr_unowned() const {
    static const char* context = "as_raw_ptr_unowned";
    ensure_compatible_rtti_held<T>(context);
    return static_cast<T*>(vptr.get());
  }

  template <typename T>
  T& lvalue_ref() const {
    static const char* context = "lvalue_ref";
    ensure_compatible_rtti_held<T>(context);
    ensure_has_pointee(context);
    return *static_cast<T*>(vptr.get());
  }

  template <typename T>
  T&& rvalue_ref() const {
    static const char* context = "rvalue_ref";
    ensure_compatible_rtti_held<T>(context);
    ensure_has_pointee(context);
    return std::move(*static_cast<T*>(vptr.get()));
  }

  template <typename T>
  static smart_holder from_raw_ptr_take_ownership(T* raw_ptr) {
    smart_holder hld;
    hld.rtti_held = &typeid(T);
    hld.vptr_deleter_guard_flag = true;
    hld.vptr_is_using_builtin_delete = true;
    hld.vptr.reset(raw_ptr,
                   guarded_builtin_delete<T>(&hld.vptr_deleter_guard_flag));
    return hld;
  }

  template <typename T>
  T* as_raw_ptr_release_ownership(
      const char* context = "as_raw_ptr_release_ownership") {
    ensure_compatible_rtti_held<T>(context);
    ensure_vptr_is_using_builtin_delete(context);
    ensure_use_count_1(context);
    T* raw_ptr = static_cast<T*>(vptr.get());
    vptr_deleter_guard_flag = false;
    vptr.reset();
    return raw_ptr;
  }

  template <typename T>
  static smart_holder from_unique_ptr(std::unique_ptr<T>&& unq_ptr) {
    smart_holder hld;
    hld.rtti_held = &typeid(T);
    hld.vptr_deleter_guard_flag = true;
    hld.vptr_is_using_builtin_delete = true;
    hld.vptr.reset(unq_ptr.get(),
                   guarded_builtin_delete<T>(&hld.vptr_deleter_guard_flag));
    unq_ptr.release();
    return hld;
  }

  template <typename T>
  std::unique_ptr<T> as_unique_ptr() {
    return std::unique_ptr<T>(as_raw_ptr_release_ownership<T>("as_unique_ptr"));
  }

  template <typename T, typename D>
  static smart_holder from_unique_ptr_with_deleter(
      std::unique_ptr<T, D>&& unq_ptr) {
    smart_holder hld;
    hld.rtti_held = &typeid(T);
    hld.rtti_uqp_del = &typeid(D);
    hld.vptr_deleter_guard_flag = true;
    hld.vptr.reset(unq_ptr.get(),
                   guarded_custom_deleter<T, D>(&hld.vptr_deleter_guard_flag));
    unq_ptr.release();
    return hld;
  }

  template <typename T, typename D>
  std::unique_ptr<T, D> as_unique_ptr_with_deleter() {
    static const char* context = "as_unique_ptr_with_deleter";
    ensure_compatible_rtti_held<T>(context);
    ensure_compatible_rtti_uqp_del<D>(context);
    ensure_use_count_1(context);
    T* raw_ptr = static_cast<T*>(vptr.get());
    vptr_deleter_guard_flag = false;
    vptr.reset();
    return std::unique_ptr<T, D>(raw_ptr);
  }

  template <typename T>
  static smart_holder from_shared_ptr(std::shared_ptr<T> shd_ptr) {
    smart_holder hld;
    hld.rtti_held = &typeid(T);
    hld.vptr_is_external_shared_ptr = true;
    hld.vptr = std::static_pointer_cast<void>(shd_ptr);
    return hld;
  }

  template <typename T>
  std::shared_ptr<T> as_shared_ptr() const {
    static const char* context = "as_shared_ptr";
    ensure_compatible_rtti_held<T>(context);
    return std::static_pointer_cast<T>(vptr);
  }
};

}  // namespace memory
}  // namespace pybindit
