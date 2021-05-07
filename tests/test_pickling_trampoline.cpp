// Copyright (c) 2021 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pybind11_tests.h"

#include <memory>
#include <utility>

namespace {

struct SimpleBase {
    int num               = 0;
    virtual ~SimpleBase() = default;

    // For compatibility with old clang versions:
    SimpleBase()                   = default;
    SimpleBase(const SimpleBase &) = default;
};

struct SimpleBaseTrampoline : SimpleBase {};

struct SimpleCppDerived : SimpleBase {};

} // namespace

TEST_SUBMODULE(pickling_trampoline, m) {
    py::class_<SimpleBase, SimpleBaseTrampoline>(m, "SimpleBase")
        .def(py::init<>())
        .def_readwrite("num", &SimpleBase::num)
        .def(py::pickle(
            [](py::object self) {
                py::dict d;
                if (py::hasattr(self, "__dict__"))
                    d = self.attr("__dict__");
                return py::make_tuple(self.attr("num"), d);
            },
            [](py::tuple t) {
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state!");
                auto cpp_state = std::unique_ptr<SimpleBase>(new SimpleBaseTrampoline);
                cpp_state->num = t[0].cast<int>();
                auto py_state  = t[1].cast<py::dict>();
                return std::make_pair(std::move(cpp_state), py_state);
            }));

    m.def("make_SimpleCppDerivedAsBase",
          []() { return std::unique_ptr<SimpleBase>(new SimpleCppDerived); });
}
