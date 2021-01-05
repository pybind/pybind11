#pragma once

#include <memory>
#include <typeinfo>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

struct smart_holder {
  std::shared_ptr<void> vptr;
  const std::type_info* rtti_held;
  const std::type_info* rtti_uqp_del;
  bool vptr_deleter_flag;

  template <typename T>
  struct vptr_deleter {
    bool* flag_ptr;
    explicit vptr_deleter(bool* flag_ptr_) : flag_ptr{flag_ptr_} {}
    void operator()(T* raw_ptr) {
      if (*flag_ptr) delete raw_ptr;
    }
  };

  smart_holder()
      : rtti_held{nullptr}, rtti_uqp_del{nullptr}, vptr_deleter_flag{false} {}

  template <typename T>
  void ensure_compatible_rtti(const char* context) {
    const std::type_info* rtti_requested = &typeid(T);
    if (!(*rtti_requested == *rtti_held)) {
      throw std::runtime_error(std::string("Incompatible RTTI (") + context +
                               ").");
    }
  }

  void ensure_use_count_1(const char* context) {
    if (vptr.use_count() != 1) {
      throw std::runtime_error(std::string("Cannot disown use_count != 1 (") +
                               context + ").");
    }
  }

  void ensure_vptr_deleter_flag_true(const char* context) {
    if (rtti_uqp_del != nullptr) {
      throw std::runtime_error(std::string("Cannot disown this shared_ptr (") +
                               context + ").");
    }
  }

  template <typename T>
  void from_raw_ptr_owned(T* raw_ptr) {
    vptr_deleter_flag = true;
    vptr.reset(raw_ptr, vptr_deleter<T>(&vptr_deleter_flag));
    rtti_held = &typeid(T);
  }

  template <typename T>
  T* as_raw_ptr_owned() {
    static const char* context = "as_raw_ptr_owned";
    ensure_compatible_rtti<T>(context);
    ensure_use_count_1(context);
    ensure_vptr_deleter_flag_true(context);
    std::shared_ptr<T> tptr = std::static_pointer_cast<T>(vptr);
    vptr.reset();
    T* result = tptr.get();
    vptr_deleter_flag = false;
    return result;
  }

  template <typename T>
  std::shared_ptr<T> as_shared_ptr() {
    static const char* context = "as_shared_ptr";
    ensure_compatible_rtti<T>(context);
    return std::static_pointer_cast<T>(vptr);
  }
};

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
