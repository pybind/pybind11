#pragma once

#include <memory>
#include <typeinfo>

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

  void clear() {
    vptr.reset();
    vptr_deleter_guard_flag = false;
    rtti_held = nullptr;
    rtti_uqp_del = nullptr;
  }

  smart_holder()
      : rtti_held{nullptr},
        rtti_uqp_del{nullptr},
        vptr_deleter_guard_flag{false} {}

  template <typename T>
  void ensure_compatible_rtti_held(const char* context) {
    const std::type_info* rtti_requested = &typeid(T);
    if (!(*rtti_requested == *rtti_held)) {
      throw std::runtime_error(std::string("Incompatible RTTI (") + context +
                               ").");
    }
  }

  template <typename D>
  void ensure_compatible_rtti_uqp_del(const char* context) {
    const std::type_info* rtti_requested = &typeid(D);
    if (!(*rtti_requested == *rtti_uqp_del)) {
      throw std::runtime_error(
          std::string("Incompatible unique_ptr deleter (") + context + ").");
    }
  }

  void ensure_vptr_deleter_guard_flag_true(const char* context) {
    if (rtti_uqp_del != nullptr) {
      throw std::runtime_error(std::string("Cannot disown this shared_ptr (") +
                               context + ").");
    }
  }

  void ensure_use_count_1(const char* context) {
    if (vptr.use_count() != 1) {
      throw std::runtime_error(std::string("Cannot disown use_count != 1 (") +
                               context + ").");
    }
  }

  template <typename T>
  void from_raw_ptr_owned(T* raw_ptr) {
    clear();
    rtti_held = &typeid(T);
    vptr_deleter_guard_flag = true;
    vptr.reset(raw_ptr, guarded_builtin_delete<T>(&vptr_deleter_guard_flag));
  }

  template <typename T>
  void from_raw_ptr_unowned(T* raw_ptr) {
    clear();
    rtti_held = &typeid(T);
    vptr_deleter_guard_flag = false;
    vptr.reset(raw_ptr, guarded_builtin_delete<T>(&vptr_deleter_guard_flag));
  }

  template <typename T>
  T* as_raw_ptr_owned(const char* context = "as_raw_ptr_owned") {
    ensure_compatible_rtti_held<T>(context);
    ensure_vptr_deleter_guard_flag_true(context);
    ensure_use_count_1(context);
    T* raw_ptr = static_cast<T*>(vptr.get());
    vptr_deleter_guard_flag = false;
    vptr.reset();
    return raw_ptr;
  }

  template <typename T>
  T* as_raw_ptr_unowned() {
    static const char* context = "as_raw_ptr_unowned";
    ensure_compatible_rtti_held<T>(context);
    return static_cast<T*>(vptr.get());
  }

  template <typename T>
  void from_unique_ptr(std::unique_ptr<T>&& unq_ptr) {
    clear();
    rtti_held = &typeid(T);
    vptr_deleter_guard_flag = true;
    vptr.reset(unq_ptr.get(),
               guarded_builtin_delete<T>(&vptr_deleter_guard_flag));
    unq_ptr.release();
  }

  template <typename T>
  std::unique_ptr<T> as_unique_ptr() {
    return std::unique_ptr<T>(as_raw_ptr_owned<T>("as_unique_ptr"));
  }

  template <typename T, typename D>
  void from_unique_ptr_with_deleter(std::unique_ptr<T, D>&& unq_ptr) {
    clear();
    rtti_held = &typeid(T);
    rtti_uqp_del = &typeid(D);
    vptr_deleter_guard_flag = true;
    vptr.reset(unq_ptr.get(),
               guarded_custom_deleter<T, D>(&vptr_deleter_guard_flag));
    unq_ptr.release();
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
  void from_shared_ptr(std::shared_ptr<T> shd_ptr) {
    clear();
    rtti_held = &typeid(T);
    vptr = std::static_pointer_cast<void>(shd_ptr);
  }

  template <typename T>
  std::shared_ptr<T> as_shared_ptr() {
    static const char* context = "as_shared_ptr";
    ensure_compatible_rtti_held<T>(context);
    return std::static_pointer_cast<T>(vptr);
  }
};

}  // namespace memory
}  // namespace pybindit
