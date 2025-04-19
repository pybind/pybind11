// Copyright (c) 2025 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pybind11_tests.h"

#include <memory>

namespace pybind11_tests {
namespace class_sp_trampoline_weak_ptr {

struct VirtBase {
    virtual ~VirtBase() = default;
    virtual int get_code() { return 100; }
};

struct PyVirtBase : VirtBase, py::trampoline_self_life_support {
    using VirtBase::VirtBase;
    int get_code() override { PYBIND11_OVERRIDE(int, VirtBase, get_code); }
};

struct WpOwner {
    void set_wp(const std::shared_ptr<VirtBase> &sp) { wp = sp; }

    int get_code() {
        auto sp = wp.lock();
        if (!sp) {
            return -999;
        }
        return sp->get_code();
    }

private:
    std::weak_ptr<VirtBase> wp;
};

} // namespace class_sp_trampoline_weak_ptr
} // namespace pybind11_tests

using namespace pybind11_tests::class_sp_trampoline_weak_ptr;

TEST_SUBMODULE(class_sp_trampoline_weak_ptr, m) {
    py::class_<VirtBase, std::shared_ptr<VirtBase>, PyVirtBase>(m, "VirtBase")
        .def(py::init<>())
        .def("get_code", &VirtBase::get_code);

    py::class_<WpOwner>(m, "WpOwner")
        .def(py::init<>())
        .def("set_wp", &WpOwner::set_wp)
        .def("get_code", &WpOwner::get_code);
}
