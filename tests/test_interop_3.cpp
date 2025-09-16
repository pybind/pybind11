/*
    tests/test_interop_3.cpp -- cross-framework interoperability tests

    Copyright (c) 2025 Hudson River Trading LLC <opensource@hudson-trading.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

// Use an unrealistically large internals version to isolate the test_interop
// modules from each other and from the rest of the pybind11 tests
#define PYBIND11_INTERNALS_VERSION 300

#include <pybind11/contrib/pymetabind.h>
#include <pybind11/pybind11.h>

#include "test_interop.h"

namespace py = pybind11;

// The following is a manual binding to `struct Shared`, created using the
// CPython C API only.

struct raw_shared_instance {
    PyObject_HEAD
    uintptr_t spacer[2]; // ensure instance layout differs from nanobind's
    bool destroy;
    bool deallocate;
    Shared *ptr;
    Shared value;
    PyObject *weakrefs;
};

static void Shared_dealloc(struct raw_shared_instance *self) {
    if (self->spacer[0] != 0x5a5a5a5a || self->spacer[1] != 0xa5a5a5a5)
        std::terminate(); // instance corrupted
    if (self->weakrefs)
        PyObject_ClearWeakRefs((PyObject *) self);
    if (self->destroy)
        self->ptr->~Shared();
    if (self->deallocate)
        free(self->ptr);

    PyTypeObject *tp = Py_TYPE((PyObject *) self);
    PyObject_Free(self);
    Py_DECREF(tp);
}

static PyObject *Shared_new(PyTypeObject *type, Shared *value, pymb_rv_policy rvp) {
    struct raw_shared_instance *self;
    self = PyObject_New(raw_shared_instance, type);
    if (self) {
        memset((char *) self + sizeof(PyObject), 0, sizeof(*self) - sizeof(PyObject));
        self->spacer[0] = 0x5a5a5a5a;
        self->spacer[1] = 0xa5a5a5a5;
        switch (rvp) {
            case pymb_rv_policy_take_ownership:
                self->ptr = value;
                self->deallocate = self->destroy = true;
                break;
            case pymb_rv_policy_copy:
                new (&self->value) Shared(*value);
                self->ptr = &self->value;
                self->deallocate = false;
                self->destroy = true;
                break;
            case pymb_rv_policy_move:
                new (&self->value) Shared(std::move(*value));
                self->ptr = &self->value;
                self->deallocate = false;
                self->destroy = true;
                break;
            case pymb_rv_policy_reference:
            case pymb_rv_policy_share_ownership:
                self->ptr = value;
                self->deallocate = self->destroy = false;
                break;
            default:
                std::terminate(); // unhandled rvp
        }
    }
    return (PyObject *) self;
}

static int Shared_init(struct raw_shared_instance *, PyObject *, PyObject *) {
    PyErr_SetString(PyExc_TypeError, "cannot be constructed from Python");
    return -1;
}

// And a minimal implementation for our "foreign framework" of the pymetabind
// interface, so nanobind can use raw_shared_instances.

static void *hook_from_python(pymb_binding *binding,
                              PyObject *pyobj,
                              uint8_t,
                              void (*)(void *ctx, PyObject *obj),
                              void *) noexcept {
    if (binding->pytype != Py_TYPE(pyobj))
        return nullptr;
    return ((raw_shared_instance *) pyobj)->ptr;
}

static PyObject *hook_to_python(pymb_binding *binding,
                                void *cobj,
                                enum pymb_rv_policy rvp,
                                pymb_to_python_feedback *feedback) noexcept {
    feedback->relocate = 0;
    if (rvp == pymb_rv_policy_none)
        return nullptr;
    feedback->is_new = 1;
    return Shared_new(binding->pytype, (Shared *) cobj, rvp);
}

static void hook_ignore_foreign_binding(pymb_binding *) noexcept {}
static void hook_ignore_foreign_framework(pymb_framework *) noexcept {}

PYBIND11_MODULE(test_interop_3, m, py::mod_gil_not_used()) {
    static PyMemberDef Shared_members[] = {
        {"__weaklistoffset__",
         Py_T_PYSSIZET,
         offsetof(struct raw_shared_instance, weakrefs),
         Py_READONLY,
         nullptr},
        {nullptr, 0, 0, 0, nullptr},
    };
    static PyType_Slot Shared_slots[] = {
        {Py_tp_doc, (void *) "Shared object"},
        {Py_tp_init, (void *) Shared_init},
        {Py_tp_dealloc, (void *) Shared_dealloc},
        {Py_tp_members, (void *) Shared_members},
        {0, nullptr},
    };
    static PyType_Spec Shared_spec = {
        /* name */ "test_interop_3.RawShared",
        /* basicsize */ sizeof(struct raw_shared_instance),
        /* itemsize */ 0,
        /* flags */ Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        /* slots */ Shared_slots,
    };

    static auto *registry = pymb_get_registry();
    if (!registry)
        throw py::error_already_set();

    static auto *fw = new pymb_framework{};
    fw->name = "example framework for pybind11 tests";
    fw->flags = pymb_framework_leak_safe;
    fw->abi_lang = pymb_abi_lang_c;
    fw->from_python = hook_from_python;
    fw->to_python = hook_to_python;
    fw->keep_alive = [](PyObject *, void *, void (*)(void *)) noexcept { return 0; };
    fw->remove_local_binding = [](pymb_binding *) noexcept {};
    fw->free_local_binding = [](pymb_binding *binding) noexcept { delete binding; };
    fw->add_foreign_binding = hook_ignore_foreign_binding;
    fw->remove_foreign_binding = hook_ignore_foreign_binding;
    fw->add_foreign_framework = hook_ignore_foreign_framework;
    fw->remove_foreign_framework = hook_ignore_foreign_framework;

    pymb_add_framework(registry, fw);
    int res = Py_AtExit(+[]() {
        pymb_remove_framework(fw);
        delete fw;
    });
    if (res != 0)
        throw py::error_already_set();

    Shared::bind_funcs</*SmartHolder=*/true>(m);
    m.def("bind_types", [hm = py::handle(m)]() { Shared::bind_types</*SmartHolder=*/true>(hm); });

    m.def("export_raw_binding", [hm = py::handle(m)]() {
        auto type = hm.attr("RawShared");
        auto *binding = new pymb_binding{};
        binding->framework = fw;
        binding->pytype = (PyTypeObject *) type.ptr();
        binding->source_name = "RawShared";
        pymb_add_binding(binding, /* tp_finalize_will_remove */ 0);
        py::import_for_interop<Shared>(type);
    });

    m.def("create_raw_binding", [hm = py::handle(m)]() {
        auto *type = (PyTypeObject *) PyType_FromSpec(&Shared_spec);
        if (!type)
            throw py::error_already_set();
#if PY_VERSION_HEX < 0x03090000
        // __weaklistoffset__ member wasn't parsed until 3.9
        type->tp_weaklistoffset = offsetof(struct raw_shared_instance, weakrefs);
#endif
        hm.attr("RawShared") = py::reinterpret_steal<py::object>((PyObject *) type);
        hm.attr("export_raw_binding")();
    });

    m.def("clear_interop_bindings", [hm = py::handle(m)]() {
        // NB: this is not a general purpose solution; the bindings removed
        // here won't be re-added if `import_all` is called
        py::list bound;
        pymb_lock_registry(registry);
        PYMB_LIST_FOREACH(struct pymb_binding *, binding, registry->bindings) {
            bound.append(py::reinterpret_borrow<py::object>((PyObject *) binding->pytype));
        }
        pymb_unlock_registry(registry);
        for (auto type : bound) {
            py::delattr(type, "__pymetabind_binding__");
        }

        bool bindings_removed = false;
        for (int i = 0; i < 5; ++i) {
            pymb_lock_registry(registry);
            bindings_removed = pymb_list_is_empty(&registry->bindings);
            pymb_unlock_registry(registry);
            if (bindings_removed) {
                break;
            }
            py::module_::import("gc").attr("collect")();
        }
        if (!bindings_removed) {
            throw std::runtime_error("Could not remove bindings");
        }

        // Restore the ability for our own create_shared() etc to work
        // properly, since that's a foreign type relationship too
        if (py::hasattr(hm, "RawShared")) {
            hm.attr("export_raw_binding")();
            py::import_for_interop<Shared>(hm.attr("RawShared"));
        }
    });
}
