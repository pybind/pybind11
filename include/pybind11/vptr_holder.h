#pragma once

#include <pybind11/pybind11.h>

#include <memory>
#include <variant>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

// Could this be a holder for a `class_`-like `vclass`?
// To enable passing of unique_ptr as in pure C++.
template <typename T> class vptr {
  public:
    explicit vptr(T *ptr = nullptr) : vptr_{std::unique_ptr<T>(ptr)} {}
    explicit vptr(std::unique_ptr<T> u) : vptr_{std::move(u)} {}
    explicit vptr(std::shared_ptr<T> s) : vptr_{s} {}

    int ownership_type() const {
        if (std::get_if<0>(&vptr_)) {
            return 0;
        }
        if (std::get_if<1>(&vptr_)) {
            return 1;
        }
        return -1;
    }

    T *get() {
        auto u = std::get_if<0>(&vptr_);
        if (u) {
            return u->get();
        }
        auto s = std::get_if<1>(&vptr_);
        if (s) {
            return s->get();
        }
        return nullptr;
    }

    std::unique_ptr<T> get_unique() {
        auto u = std::get_if<0>(&vptr_);
        if (u) {
            return std::move(*u);
        }
        throw std::runtime_error("get_unique failure.");
    }

    std::shared_ptr<T> get_shared() {
        auto s = std::get_if<1>(&vptr_);
        if (s) {
            return *s;
        }
        auto u = std::get_if<0>(&vptr_);
        if (u) {
            auto result = std::shared_ptr<T>(std::move(*u));
            vptr_ = result;
            return result;
        }
        throw std::runtime_error("get_shared failure.");
    }

  private:
    std::variant<std::unique_ptr<T>, std::shared_ptr<T>> vptr_;
};

template <typename T> class vptr_holder : public vptr<T> {
    using vptr<T>::vptr;
};

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

PYBIND11_DECLARE_HOLDER_TYPE(T, pybind11::vptr_holder<T>);
