// Copyright (c) 2021 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pybind11/trampoline_self_life_support.h"
#include "pybind11_tests.h"

#include <cstdint>

namespace pybind11_tests {
namespace class_sh_trampoline_unique_ptr {

class Class {
public:
    virtual ~Class() = default;

    void setVal(std::uint64_t val) { val_ = val; }
    std::uint64_t getVal() const { return val_; }

    virtual std::unique_ptr<Class> clone() const = 0;
    virtual int foo() const = 0;

protected:
    Class() = default;

    // Some compilers complain about implicitly defined versions of some of the following:
    Class(const Class &) = default;

private:
    std::uint64_t val_ = 0;
};

} // namespace class_sh_trampoline_unique_ptr
} // namespace pybind11_tests

namespace pybind11_tests {
namespace class_sh_trampoline_unique_ptr {

class PyClass : public Class, public py::trampoline_self_life_support {
public:
    std::unique_ptr<Class> clone() const override {
        PYBIND11_OVERRIDE_PURE(std::unique_ptr<Class>, Class, clone);
    }

    int foo() const override { PYBIND11_OVERRIDE_PURE(int, Class, foo); }
};

} // namespace class_sh_trampoline_unique_ptr
} // namespace pybind11_tests

TEST_SUBMODULE(class_sh_trampoline_unique_ptr, m) {
    using namespace pybind11_tests::class_sh_trampoline_unique_ptr;

    py::classh<Class, PyClass>(m, "Class")
        .def(py::init<>())
        .def("set_val", &Class::setVal)
        .def("get_val", &Class::getVal)
        .def("clone", &Class::clone)
        .def("foo", &Class::foo);

    m.def("clone", [](const Class &obj) { return obj.clone(); });
    m.def("clone_and_foo", [](const Class &obj) { return obj.clone()->foo(); });
}
