#include "pybind11/cast.h"
#include "pybind11_tests.h"

#include <memory>
#include <string>

namespace const_only_smart_ptr {

template <class T>
class non_sync_const_shared_ptr {
public:
    explicit non_sync_const_shared_ptr(const T *ptr) {
        try {
            counter_ = new uint64_t(1);
        } catch (...) {
            delete ptr;
        }

        ptr_ = ptr;
    }

    non_sync_const_shared_ptr(const non_sync_const_shared_ptr &other)
        : ptr_(other.ptr_), counter_(other.counter_) {
        ++*counter_;
    }

    non_sync_const_shared_ptr(non_sync_const_shared_ptr &&other) noexcept
        : ptr_(other.ptr_), counter_(other.counter_) {
        other.ptr_ = nullptr;
        other.counter_ = nullptr;
    }

    ~non_sync_const_shared_ptr() {
        if (!counter_) {
            return;
        }

        --*counter_;

        if (*counter_ == 0) {
            delete ptr_;
        }
    }

    const T *get() const { return ptr_; }

    const T &operator*() const { return *ptr_; }
    const T *operator->() const { return ptr_; }

private:
    const T *ptr_ = nullptr;
    uint64_t *counter_ = nullptr;
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

PYBIND11_DECLARE_HOLDER_TYPE(T, const_only_smart_ptr::non_sync_const_shared_ptr<T>, true)

TEST_SUBMODULE(const_module, m) {
    py::class_<const_only_smart_ptr::MyData,
               const_only_smart_ptr::non_sync_const_shared_ptr<const_only_smart_ptr::MyData>>(
        m, "Data")
        .def(py::init(
            [](const std::string &name) { return const_only_smart_ptr::MyData::create(name); }))
        .def_property_readonly("name", &const_only_smart_ptr::MyData::getName);
}
