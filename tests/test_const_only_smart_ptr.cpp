#include "pybind11/cast.h"
#include "pybind11_tests.h"

#include <memory>
#include <string>

namespace const_only_smart_ptr {

template <class T>
class const_only_shared_ptr {
public:
    const_only_shared_ptr() = default;

    explicit const_only_shared_ptr(const T *ptr) : ptr_(ptr) {}

    const_only_shared_ptr(const const_only_shared_ptr &) = default;
    const_only_shared_ptr(const_only_shared_ptr &&) = default;

    const_only_shared_ptr &operator=(const const_only_shared_ptr &) = default;
    const_only_shared_ptr &operator=(const_only_shared_ptr &&) = default;

    ~const_only_shared_ptr() = default;

    const T *get() const { return ptr_.get(); }
    const T &operator*() const { return *ptr_; }
    const T *operator->() const { return ptr_.get(); }
    explicit operator bool() const { return ptr_ != nullptr; }

private:
    // for demonstration purpose only, this imitates smart pointer with a const-only pointer
    std::shared_ptr<const T> ptr_;
};

class MyData {
public:
    static const_only_shared_ptr<MyData> create(std::string name) {
        return const_only_shared_ptr<MyData>(new MyData(std::move(name)));
    }

    const std::string &getName() const { return name_; }

private:
    explicit MyData(std::string &&name) : name_(std::move(name)) {}

    std::string name_;
};
} // namespace const_only_smart_ptr

using namespace const_only_smart_ptr;

PYBIND11_DECLARE_HOLDER_TYPE(T, const_only_shared_ptr<T>, true)

TEST_SUBMODULE(const_module, m) {
    py::class_<MyData, const_only_shared_ptr<MyData>>(m, "Data")
        .def(py::init([](const std::string &name) { return MyData::create(name); }))
        .def_property_readonly("name", &MyData::getName);
}
