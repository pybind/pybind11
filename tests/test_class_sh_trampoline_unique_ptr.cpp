// Copyright (c) 2021 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pybind11/smart_holder.h"
#include "pybind11/virtual_overrider_self_life_support.h"
#include "pybind11_tests.h"

namespace {

class Class {
public:
    virtual ~Class()                             = default;
    virtual std::unique_ptr<Class> clone() const = 0;
    virtual int foo() const                      = 0;

protected:
    Class() = default;

    // Some compilers complain about implicitly defined versions of some of the following:
    Class(const Class &) = default;
};

} // namespace

PYBIND11_SMART_HOLDER_TYPE_CASTERS(Class)

namespace {

class PyClass : public Class, public py::virtual_overrider_self_life_support {
public:
    std::unique_ptr<Class> clone() const override {
        PYBIND11_OVERRIDE_PURE(std::unique_ptr<Class>, Class, clone);
    }

    int foo() const override { PYBIND11_OVERRIDE_PURE(int, Class, foo); }
};

} // namespace

TEST_SUBMODULE(class_sh_trampoline_unique_ptr, m) {
    py::classh<Class, PyClass>(m, "Class")
        .def(py::init<>())
        .def("clone", &Class::clone)
        .def("foo", &Class::foo);

    m.def("clone_and_foo", [](const Class &obj) { return obj.clone()->foo(); });
}
