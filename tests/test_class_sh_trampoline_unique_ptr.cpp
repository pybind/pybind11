// Copyright (c) 2021 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pybind11/smart_holder.h"
#include "pybind11/trampoline_self_life_support.h"
#include "pybind11_tests.h"

#include <cstdint>

namespace pybind11_tests {
namespace class_sh_trampoline_basic {

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

#ifdef PYBIND11_HAVE_INTERNALS_WITH_SMART_HOLDER_SUPPORT
class PyClass : public Class, public py::trampoline_self_life_support {
public:
    std::unique_ptr<Class> clone() const override {
        PYBIND11_OVERRIDE_PURE(std::unique_ptr<Class>, Class, clone);
    }

    int foo() const override { PYBIND11_OVERRIDE_PURE(int, Class, foo); }
};
#endif

} // namespace class_sh_trampoline_basic
} // namespace pybind11_tests

using namespace pybind11_tests::class_sh_trampoline_basic;

PYBIND11_SMART_HOLDER_TYPE_CASTERS(Class)

TEST_SUBMODULE(class_sh_trampoline_unique_ptr, m) {
    m.attr("defined_PYBIND11_HAVE_INTERNALS_WITH_SMART_HOLDER_SUPPORT") =
#ifndef PYBIND11_HAVE_INTERNALS_WITH_SMART_HOLDER_SUPPORT
        false;
#else
        true;

    py::classh<Class, PyClass>(m, "Class")
        .def(py::init<>())
        .def("set_val", &Class::setVal)
        .def("get_val", &Class::getVal)
        .def("clone", &Class::clone)
        .def("foo", &Class::foo);

    m.def("clone", [](const Class &obj) { return obj.clone(); });
    m.def("clone_and_foo", [](const Class &obj) { return obj.clone()->foo(); });
#endif // PYBIND11_HAVE_INTERNALS_WITH_SMART_HOLDER_SUPPORT
}
