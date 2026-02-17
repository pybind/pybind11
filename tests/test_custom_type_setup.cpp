/*
    tests/test_custom_type_setup.cpp -- Tests `pybind11::custom_type_setup`

    Copyright (c) Google LLC

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <pybind11/detail/internals.h>
#include <pybind11/pybind11.h>

#include "pybind11_tests.h"

#include <vector>

namespace py = pybind11;

namespace {
struct ContainerOwnsPythonObjects {
    std::vector<py::object> list;

    void append(const py::object &obj) { list.emplace_back(obj); }
    py::object at(py::ssize_t index) const {
        if (index >= size() || index < 0) {
            throw py::index_error("Index out of range");
        }
        return list.at(py::size_t(index));
    }
    py::ssize_t size() const { return py::ssize_t_cast(list.size()); }
    void clear() { list.clear(); }
};

void add_gc_checkers_with_weakrefs(const py::object &obj) {
    py::handle global_capsule = py::detail::get_internals_capsule();
    if (!global_capsule) {
        throw std::runtime_error("No global internals capsule found");
    }
    (void) py::weakref(obj, py::cpp_function([global_capsule](py::handle weakref) -> void {
                           py::handle current_global_capsule = py::detail::get_internals_capsule();
                           if (!current_global_capsule.is(global_capsule)) {
                               throw std::runtime_error(
                                   "Global internals capsule was destroyed prematurely");
                           }
                           weakref.dec_ref();
                       }))
        .release();

    py::handle local_capsule = py::detail::get_local_internals_capsule();
    if (!local_capsule) {
        throw std::runtime_error("No local internals capsule found");
    }
    (void) py::weakref(
        obj, py::cpp_function([local_capsule](py::handle weakref) -> void {
            py::handle current_local_capsule = py::detail::get_local_internals_capsule();
            if (!current_local_capsule.is(local_capsule)) {
                throw std::runtime_error("Local internals capsule was destroyed prematurely");
            }
            weakref.dec_ref();
        }))
        .release();
}
} // namespace

TEST_SUBMODULE(custom_type_setup, m) {
    py::class_<ContainerOwnsPythonObjects> cls(
        m,
        "ContainerOwnsPythonObjects",
        // Please review/update docs/advanced/classes.rst after making changes here.
        py::custom_type_setup([](PyHeapTypeObject *heap_type) {
            auto *type = &heap_type->ht_type;
            type->tp_flags |= Py_TPFLAGS_HAVE_GC;
            type->tp_traverse = [](PyObject *self_base, visitproc visit, void *arg) {
// https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject.tp_traverse
#if PY_VERSION_HEX >= 0x03090000
                Py_VISIT(Py_TYPE(self_base));
#endif
                if (py::detail::is_holder_constructed(self_base)) {
                    auto &self = py::cast<ContainerOwnsPythonObjects &>(py::handle(self_base));
                    for (auto &item : self.list) {
                        Py_VISIT(item.ptr());
                    }
                }
                return 0;
            };
            type->tp_clear = [](PyObject *self_base) {
                if (py::detail::is_holder_constructed(self_base)) {
                    auto &self = py::cast<ContainerOwnsPythonObjects &>(py::handle(self_base));
                    for (auto &item : self.list) {
                        Py_CLEAR(item.ptr());
                    }
                    self.list.clear();
                }
                return 0;
            };
        }));
    cls.def(py::init<>());
    cls.def("append", &ContainerOwnsPythonObjects::append);
    cls.def("at", &ContainerOwnsPythonObjects::at);
    cls.def("size", &ContainerOwnsPythonObjects::size);
    cls.def("clear", &ContainerOwnsPythonObjects::clear);

    m.def("add_gc_checkers_with_weakrefs", &add_gc_checkers_with_weakrefs);
}
