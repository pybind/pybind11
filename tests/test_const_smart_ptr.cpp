#include "pybind11/cast.h"
#include "pybind11_tests.h"

#include <memory>
#include <string>
#include <utility>

template <class T>
class SingleThreadedSharedPtr {
public:
    explicit SingleThreadedSharedPtr(const T *ptr) {
        try {
            counter_ = new uint64_t(1);
        } catch (...) {
            delete ptr;
        }

        ptr_ = ptr;
    }

    SingleThreadedSharedPtr(const SingleThreadedSharedPtr &other)
        : ptr_(other.ptr_), counter_(other.counter_) {
        ++*counter_;
    }

    SingleThreadedSharedPtr(SingleThreadedSharedPtr &&other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)),
          counter_(std::exchange(other.counter_, nullptr)) {}

    ~SingleThreadedSharedPtr() {
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

PYBIND11_DECLARE_HOLDER_TYPE(T, SingleThreadedSharedPtr<T>, true)

class MyData {
public:
    static SingleThreadedSharedPtr<MyData> create(std::string name) {
        return SingleThreadedSharedPtr<MyData>(new MyData(std::move(name)));
    }

    const std::string &getName() const { return name_; }

private:
    explicit MyData(std::string &&name) : name_(std::move(name)) {}

    std::string name_;
};

TEST_SUBMODULE(const_module, m) {
    py::class_<MyData, SingleThreadedSharedPtr<MyData>>(m, "Data")
        .def(py::init([](const std::string &name) { return MyData::create(name); }))
        .def_property_readonly("name", &MyData::getName);
}
