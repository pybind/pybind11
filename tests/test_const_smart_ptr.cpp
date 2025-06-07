#include "pybind11/cast.h"
#include "pybind11_tests.h"

#include <string>

template <class T>
class StaticPtr {
public:
    explicit StaticPtr(const T *ptr) : ptr_(ptr) {}

    const T *get() const { return ptr_; }

    const T &operator*() const { return *ptr_; }
    const T *operator->() const { return ptr_; }

private:
    const T *ptr_ = nullptr;
};

PYBIND11_DECLARE_HOLDER_TYPE(T, StaticPtr<T>, true)

class MyData {
public:
    static StaticPtr<MyData> create(std::string name) {
        return StaticPtr(new MyData(std::move(name)));
    }

    const std::string &getName() const { return name_; }

private:
    explicit MyData(std::string &&name) : name_(std::move(name)) {}

    std::string name_;
};

TEST_SUBMODULE(const_module, m) {
    py::class_<MyData, StaticPtr<MyData>>(m, "Data")
        .def(py::init([](const std::string &name) { return MyData::create(name); }))
        .def_property_readonly("name", &MyData::getName);
}
