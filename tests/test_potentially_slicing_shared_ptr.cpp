// Copyright (c) 2025 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pybind11_tests.h"

#include <memory>

namespace pybind11_tests {
namespace potentially_slicing_shared_ptr {

template <int> // Using int as a trick to easily generate multiple types.
struct VirtBase {
    virtual ~VirtBase() = default;
    virtual int get_code() { return 100; }
};

using VirtBaseSH = VirtBase<0>; // for testing with py::smart_holder
using VirtBaseSP = VirtBase<1>; // for testing with std::shared_ptr as holder

struct PyVirtBaseSH : VirtBaseSH, py::trampoline_self_life_support {
    using VirtBaseSH::VirtBaseSH;
    int get_code() override { PYBIND11_OVERRIDE(int, VirtBaseSH, get_code); }
};

struct PyVirtBaseSP : VirtBaseSP { // self-life-support not available
    using VirtBaseSP::VirtBaseSP;
    int get_code() override { PYBIND11_OVERRIDE(int, VirtBaseSP, get_code); }
};

template <typename VB>
std::shared_ptr<VB> rtrn_obj_cast_shared_ptr(py::handle obj) {
    return obj.cast<std::shared_ptr<VB>>();
}

template <typename VB>
std::shared_ptr<VB> rtrn_potentially_slicing_shared_ptr(py::handle obj) {
    return py::potentially_slicing_shared_ptr<VB>(obj);
}

template <typename VB>
struct SpOwner {
    void set_sp(const std::shared_ptr<VB> &sp_) { sp = sp_; }

    int get_code() {
        if (!sp) {
            return -888;
        }
        return sp->get_code();
    }

private:
    std::shared_ptr<VB> sp;
};

template <typename VB>
struct WpOwner {
    void set_wp(const std::shared_ptr<VB> &sp) { wp = sp; }

    int get_code() {
        auto sp = wp.lock();
        if (!sp) {
            return -999;
        }
        return sp->get_code();
    }

private:
    std::weak_ptr<VB> wp;
};

template <typename VB>
void wrap(py::module_ &m,
          const char *roc_pyname,
          const char *rps_pyname,
          const char *spo_pyname,
          const char *wpo_pyname) {
    m.def(roc_pyname, rtrn_obj_cast_shared_ptr<VB>);
    m.def(rps_pyname, rtrn_potentially_slicing_shared_ptr<VB>);

    py::classh<SpOwner<VB>>(m, spo_pyname)
        .def(py::init<>())
        .def("set_sp", &SpOwner<VB>::set_sp)
        .def("get_code", &SpOwner<VB>::get_code);

    py::classh<WpOwner<VB>>(m, wpo_pyname)
        .def(py::init<>())
        .def("set_wp", &WpOwner<VB>::set_wp)
        .def("set_wp_potentially_slicing",
             [](WpOwner<VB> &self, py::handle obj) {
                 self.set_wp(py::potentially_slicing_shared_ptr<VB>(obj));
             })
        .def("get_code", &WpOwner<VB>::get_code);
}

} // namespace potentially_slicing_shared_ptr
} // namespace pybind11_tests

using namespace pybind11_tests::potentially_slicing_shared_ptr;

TEST_SUBMODULE(potentially_slicing_shared_ptr, m) {
    py::classh<VirtBaseSH, PyVirtBaseSH>(m, "VirtBaseSH")
        .def(py::init<>())
        .def("get_code", &VirtBaseSH::get_code);

    py::class_<VirtBaseSP, std::shared_ptr<VirtBaseSP>, PyVirtBaseSP>(m, "VirtBaseSP")
        .def(py::init<>())
        .def("get_code", &VirtBaseSP::get_code);

    wrap<VirtBaseSH>(m,
                     "SH_rtrn_obj_cast_shared_ptr",
                     "SH_rtrn_potentially_slicing_shared_ptr",
                     "SH_SpOwner",
                     "SH_WpOwner");

    wrap<VirtBaseSP>(m,
                     "SP_rtrn_obj_cast_shared_ptr",
                     "SP_rtrn_potentially_slicing_shared_ptr",
                     "SP_SpOwner",
                     "SP_WpOwner");
}
