#include "pybind11/cast.h"
#include "pybind11_tests.h"

#include <memory>
#include <string>

namespace const_only_smart_ptr {

template <class T>
class non_sync_const_shared_ptr {
public:
    non_sync_const_shared_ptr() = default;

    explicit non_sync_const_shared_ptr(const T *ptr) {
        if (ptr) {
            try {
                counter_ = new size_t(1);
            } catch (...) {
                delete ptr;
                throw;
            }
        }

        ptr_ = ptr;
    }

    non_sync_const_shared_ptr(const non_sync_const_shared_ptr &other)
        : ptr_(other.ptr_), counter_(other.counter_) {
        if (counter_) {
            ++*counter_;
        }
    }

    non_sync_const_shared_ptr(non_sync_const_shared_ptr &&other) noexcept
        : ptr_(other.ptr_), counter_(other.counter_) {
        other.ptr_ = nullptr;
        other.counter_ = nullptr;
    }

    non_sync_const_shared_ptr &operator=(const non_sync_const_shared_ptr &other) {
        if (this == &other) {
            return *this;
        }

        release();

        ptr_ = other.ptr_;
        counter_ = other.counter_;

        if (counter_) {
            ++*counter_;
        }

        return *this;
    }

    non_sync_const_shared_ptr &operator=(non_sync_const_shared_ptr &&other) noexcept {
        if (this == &other) {
            return *this;
        }

        release();

        ptr_ = other.ptr_;
        counter_ = other.counter_;

        other.ptr_ = nullptr;
        other.counter_ = nullptr;

        return *this;
    }

    ~non_sync_const_shared_ptr() { release(); }

    const T *get() const { return ptr_; }

    const T &operator*() const { return *ptr_; }
    const T *operator->() const { return ptr_; }

    explicit operator bool() const { return ptr_ != nullptr; }

private:
    void release() noexcept {
        if (!counter_) {
            return;
        }

        --*counter_;

        if (*counter_ == 0) {
            delete ptr_;
            delete counter_;
        }

        ptr_ = nullptr;
        counter_ = nullptr;
    }

    const T *ptr_ = nullptr;
    size_t *counter_ = nullptr;
};

class MyData {
public:
    static non_sync_const_shared_ptr<MyData> create(std::string name) {
        return non_sync_const_shared_ptr<MyData>(new MyData(std::move(name)));
    }

    const std::string &getName() const { return name_; }

private:
    explicit MyData(std::string &&name) : name_(std::move(name)) {}

    std::string name_;
};
} // namespace const_only_smart_ptr

using namespace const_only_smart_ptr;

PYBIND11_DECLARE_HOLDER_TYPE(T, non_sync_const_shared_ptr<T>, true)

TEST_SUBMODULE(const_module, m) {
    py::class_<MyData, non_sync_const_shared_ptr<MyData>>(m, "Data")
        .def(py::init([](const std::string &name) { return MyData::create(name); }))
        .def_property_readonly("name", &MyData::getName);
}
