// Copyright (c) 2025 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pybind11_tests.h"

#include <memory>

namespace pybind11_tests {
namespace potentially_slicing_shared_ptr {

struct VirtBase {
    virtual ~VirtBase() = default;
    virtual int get_code() { return 100; }
};

struct PyVirtBase : VirtBase, py::trampoline_self_life_support {
    using VirtBase::VirtBase;
    int get_code() override { PYBIND11_OVERRIDE(int, VirtBase, get_code); }
};

std::shared_ptr<VirtBase> rtrn_obj_cast_shared_ptr(py::handle obj) {
    return obj.cast<std::shared_ptr<VirtBase>>();
}

std::shared_ptr<VirtBase> rtrn_potentially_slicing_shared_ptr(py::handle obj) {
    return py::potentially_slicing_shared_ptr<VirtBase>(obj);
}

struct SpOwner {
    void set_sp(const std::shared_ptr<VirtBase> &sp_) { sp = sp_; }

    int get_code() {
        if (!sp) {
            return -888;
        }
        return sp->get_code();
    }

private:
    std::shared_ptr<VirtBase> sp;
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

} // namespace potentially_slicing_shared_ptr
} // namespace pybind11_tests

using namespace pybind11_tests::potentially_slicing_shared_ptr;

TEST_SUBMODULE(potentially_slicing_shared_ptr, m) {
    py::classh<VirtBase, PyVirtBase>(m, "VirtBase")
        .def(py::init<>())
        .def("get_code", &VirtBase::get_code);

    m.def("rtrn_obj_cast_shared_ptr", rtrn_obj_cast_shared_ptr);
    m.def("rtrn_potentially_slicing_shared_ptr", rtrn_potentially_slicing_shared_ptr);

    py::classh<SpOwner>(m, "SpOwner")
        .def(py::init<>())
        .def("set_sp", &SpOwner::set_sp)
        .def("get_code", &SpOwner::get_code);

    py::classh<WpOwner>(m, "WpOwner")
        .def(py::init<>())
        .def("set_wp", &WpOwner::set_wp)
        .def("set_wp_potentially_slicing",
             [](WpOwner &self, py::handle obj) {
                 self.set_wp(py::potentially_slicing_shared_ptr<VirtBase>(obj));
             })
        .def("get_code", &WpOwner::get_code);
}
