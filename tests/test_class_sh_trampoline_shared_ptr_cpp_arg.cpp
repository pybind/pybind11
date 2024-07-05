// Copyright (c) 2021 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pybind11/smart_holder.h"
#include "pybind11_tests.h"

#include <utility>

namespace {

// For testing whether a python subclass of a C++ object dies when the
// last python reference is lost
struct SpBase {
    // returns true if the base virtual function is called
    virtual bool is_base_used() { return true; }

    // returns true if there's an associated python instance
    bool has_python_instance() {
        auto *tinfo = py::detail::get_type_info(typeid(SpBase));
        return (bool) py::detail::get_object_handle(this, tinfo);
    }

    SpBase() = default;
    SpBase(const SpBase &) = delete;
    virtual ~SpBase() = default;
};

std::shared_ptr<SpBase> pass_through_shd_ptr(const std::shared_ptr<SpBase> &obj) { return obj; }

struct PySpBase : SpBase {
    using SpBase::SpBase;
    bool is_base_used() override { PYBIND11_OVERRIDE(bool, SpBase, is_base_used); }
};

struct SpBaseTester {
    std::shared_ptr<SpBase> get_object() const { return m_obj; }
    void set_object(std::shared_ptr<SpBase> obj) { m_obj = std::move(obj); }
    bool is_base_used() { return m_obj->is_base_used(); }
    bool has_instance() { return (bool) m_obj; }
    bool has_python_instance() { return m_obj && m_obj->has_python_instance(); }
    void set_nonpython_instance() { m_obj = std::make_shared<SpBase>(); }
    std::shared_ptr<SpBase> m_obj;
};

// For testing that a C++ class without an alias does not retain the python
// portion of the object
struct SpGoAway {};

struct SpGoAwayTester {
    std::shared_ptr<SpGoAway> m_obj;
};

} // namespace

PYBIND11_SMART_HOLDER_TYPE_CASTERS(SpBase)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(SpBaseTester)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(SpGoAway)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(SpGoAwayTester)

TEST_SUBMODULE(class_sh_trampoline_shared_ptr_cpp_arg, m) {
    // For testing whether a python subclass of a C++ object dies when the
    // last python reference is lost

    py::classh<SpBase, PySpBase>(m, "SpBase")
        .def(py::init<>())
        .def(py::init([](int) { return std::make_shared<PySpBase>(); }))
        .def("is_base_used", &SpBase::is_base_used)
        .def("has_python_instance", &SpBase::has_python_instance);

    m.def("pass_through_shd_ptr", pass_through_shd_ptr);
    m.def("pass_through_shd_ptr_release_gil",
          pass_through_shd_ptr,
          py::call_guard<py::gil_scoped_release>()); // PR #4196

    py::classh<SpBaseTester>(m, "SpBaseTester")
        .def(py::init<>())
        .def("get_object", &SpBaseTester::get_object)
        .def("set_object", &SpBaseTester::set_object)
        .def("is_base_used", &SpBaseTester::is_base_used)
        .def("has_instance", &SpBaseTester::has_instance)
        .def("has_python_instance", &SpBaseTester::has_python_instance)
        .def("set_nonpython_instance", &SpBaseTester::set_nonpython_instance)
        .def_readwrite("obj", &SpBaseTester::m_obj);

    // For testing that a C++ class without an alias does not retain the python
    // portion of the object

    py::classh<SpGoAway>(m, "SpGoAway").def(py::init<>());

    py::classh<SpGoAwayTester>(m, "SpGoAwayTester")
        .def(py::init<>())
        .def_readwrite("obj", &SpGoAwayTester::m_obj);
}
