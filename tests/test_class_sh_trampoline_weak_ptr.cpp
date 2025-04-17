// Copyright (c) 2021 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pybind11_tests.h"

#include <utility>

namespace pybind11_tests {
namespace class_sh_trampoline_weak_ptr {

// // For testing whether a python subclass of a C++ object can be accessed from a C++ weak_ptr
struct WpBase {
    // returns true if the base virtual function is called
    virtual bool is_base_used() { return true; }

    // returns true if there's an associated python instance
    bool has_python_instance() {
        auto *tinfo = py::detail::get_type_info(typeid(WpBase));
        return (bool) py::detail::get_object_handle(this, tinfo);
    }

    WpBase() = default;
    WpBase(const WpBase &) = delete;
    virtual ~WpBase() = default;
};

struct PyWpBase : WpBase {
    using WpBase::WpBase;
    bool is_base_used() override { PYBIND11_OVERRIDE(bool, WpBase, is_base_used); }
};

struct WpBaseTester {
    std::shared_ptr<WpBase> get_object() const { return m_obj.lock(); }
    void set_object(std::shared_ptr<WpBase> obj) { m_obj = obj; }
    bool is_expired() { return m_obj.expired(); }
    bool is_base_used() { return m_obj.lock()->is_base_used(); }
    std::weak_ptr<WpBase> m_obj;
};

} // namespace class_sh_trampoline_weak_ptr
} // namespace pybind11_tests

using namespace pybind11_tests::class_sh_trampoline_weak_ptr;

TEST_SUBMODULE(class_sh_trampoline_weak_ptr, m) {
    // For testing whether a python subclass of a C++ object can be accessed from a C++ weak_ptr

    py::classh<WpBase, PyWpBase>(m, "WpBase")
        .def(py::init<>())
        .def(py::init([](int) { return std::make_shared<PyWpBase>(); }))
        .def("is_base_used", &WpBase::is_base_used)
        .def("has_python_instance", &WpBase::has_python_instance);

    py::classh<WpBaseTester>(m, "WpBaseTester")
        .def(py::init<>())
        .def("get_object", &WpBaseTester::get_object)
        .def("set_object", &WpBaseTester::set_object)
        .def("is_expired", &WpBaseTester::is_expired)
        .def("is_base_used", &WpBaseTester::is_base_used);
}
