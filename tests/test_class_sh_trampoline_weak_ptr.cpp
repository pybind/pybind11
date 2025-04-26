// Copyright (c) 2025 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pybind11_tests.h"

#include <memory>

namespace pybind11_tests {
namespace class_sh_trampoline_weak_ptr {

struct VirtBase {
    virtual ~VirtBase() = default;
    virtual int get_code() { return 100; }
};

struct PyVirtBase : VirtBase /*, py::trampoline_self_life_support */ {
    using VirtBase::VirtBase;
    int get_code() override { PYBIND11_OVERRIDE(int, VirtBase, get_code); }

    ~PyVirtBase() override {
        fflush(stderr);
        printf("\nLOOOK ~PyVirtBase()\n");
        fflush(stdout);
    }
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

std::shared_ptr<VirtBase> pass_through_sp_VirtBase(const std::shared_ptr<VirtBase> &sp) {
    return sp;
}

} // namespace class_sh_trampoline_weak_ptr
} // namespace pybind11_tests

using namespace pybind11_tests::class_sh_trampoline_weak_ptr;

TEST_SUBMODULE(class_sh_trampoline_weak_ptr, m) {
    py::classh<VirtBase, PyVirtBase>(m, "VirtBase")
        .def(py::init<>())
        .def("get_code", &VirtBase::get_code);

    py::classh<WpOwner>(m, "WpOwner")
        .def(py::init<>())
        .def("set_wp",
             [](WpOwner &self, py::handle obj) {
                 self.set_wp(obj.cast<std::shared_ptr<VirtBase>>());
             })
        .def("get_code", &WpOwner::get_code);

    m.def("pass_through_sp_VirtBase", pass_through_sp_VirtBase);
}
