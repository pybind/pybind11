// Copyright (c) 2021 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pybind11_tests.h"

namespace {

// For testing whether a python subclass of a C++ object dies when the
// last python reference is lost
struct SpBase {
    // returns true if the base virtual function is called
    virtual bool is_base_used() { return true; }

    SpBase()               = default;
    SpBase(const SpBase &) = delete;
    virtual ~SpBase()      = default;
};

struct PySpBase : SpBase {
    bool is_base_used() override { PYBIND11_OVERRIDE(bool, SpBase, is_base_used); }
};

struct SpBaseTester {
    std::shared_ptr<SpBase> get_object() { return m_obj; }
    void set_object(std::shared_ptr<SpBase> obj) { m_obj = obj; }
    bool is_base_used() { return m_obj->is_base_used(); }
    std::shared_ptr<SpBase> m_obj;
};

// For testing that a C++ class without an alias does not retain the python
// portion of the object
struct SpGoAway {};

struct SpGoAwayTester {
    std::shared_ptr<SpGoAway> m_obj;
};

} // namespace

TEST_SUBMODULE(trampoline_shared_ptr_cpp_arg, m) {
    // For testing whether a python subclass of a C++ object dies when the
    // last python reference is lost

    py::class_<SpBase, std::shared_ptr<SpBase>, PySpBase>(m, "SpBase")
        .def(py::init<>())
        .def("is_base_used", &SpBase::is_base_used);

    py::class_<SpBaseTester>(m, "SpBaseTester")
        .def(py::init<>())
        .def("get_object", &SpBaseTester::get_object)
        .def("set_object", &SpBaseTester::set_object)
        .def("is_base_used", &SpBaseTester::is_base_used)
        .def_readwrite("obj", &SpBaseTester::m_obj);

    // For testing that a C++ class without an alias does not retain the python
    // portion of the object

    py::class_<SpGoAway, std::shared_ptr<SpGoAway>>(m, "SpGoAway").def(py::init<>());

    py::class_<SpGoAwayTester>(m, "SpGoAwayTester")
        .def(py::init<>())
        .def_readwrite("obj", &SpGoAwayTester::m_obj);
}
