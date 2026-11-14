// Copyright (c) 2024-2025 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

// For background see the description of PR google/pybind11clif#30099.

#pragma once

#include <pybind11/attr.h>
#include <pybind11/conduit/pybind11_platform_abi_id.h>
#include <pybind11/pytypes.h>

#include "common.h"

#include <cassert>
#include <cstring>
#include <utility>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

struct function_record_PyObject {
#if !defined(PYBIND11_OPAQUE_PYOBJECT)
    PyObject_HEAD
#endif
    function_record *cpp_func_rec;
};

inline function_record_PyObject *function_record_data(PyObject *obj) {
    return type_data<function_record_PyObject>(obj);
}

// Python computes the `__qualname__` of a builtin function from its `__self__`, which is the
// function record. A module gives no prefix, so the record of a function that is not in a class
// is a module subclass. The record type for the functions in a class has the `__qualname__` of
// that class.
#if !defined(PYPY_VERSION) && !defined(GRAALVM_PYTHON)
#    define PYBIND11_FUNCTION_RECORD_QUALNAME
#endif

#if defined(PYBIND11_FUNCTION_RECORD_QUALNAME)
/// The record pointer of a function record that is a module subclass.
inline function_record *&function_record_module_data(PyObject *obj) {
#    if PY_VERSION_HEX >= 0x030C0000
    return *static_cast<function_record **>(PyObject_GetTypeData(obj, Py_TYPE(obj)));
#    else
    return *reinterpret_cast<function_record **>(reinterpret_cast<char *>(obj)
                                                 + PyModule_Type.tp_basicsize);
#    endif
}
#endif

PYBIND11_NAMESPACE_BEGIN(function_record_PyTypeObject_methods)

PyObject *tp_new_impl(PyTypeObject *type, PyObject *args, PyObject *kwds);
PyObject *tp_alloc_impl(PyTypeObject *type, Py_ssize_t nitems);
int tp_init_impl(PyObject *self, PyObject *args, PyObject *kwds);
void tp_dealloc_impl(PyObject *self);
void tp_free_impl(void *self);

static PyObject *reduce_ex_impl(PyObject *self, PyObject *, PyObject *);

static PyMethodDef tp_methods_impl[]
    = {{"__reduce_ex__",
        // reduce_ex_impl is a PyCFunctionWithKeywords, but PyMethodDef
        // requires a PyCFunction. The cast through void* is safe and
        // idiomatic with METH_KEYWORDS, and it successfully sidesteps
        // unhelpful compiler warnings.
        // NOLINTNEXTLINE(bugprone-casting-through-void)
        reinterpret_cast<PyCFunction>(reinterpret_cast<void *>(reduce_ex_impl)),
        METH_VARARGS | METH_KEYWORDS,
        nullptr},
       {nullptr, nullptr, 0, nullptr}};

// Python 3.12+ emits a DeprecationWarning for heap types whose tp_name does
// not contain a dot ('.') and that lack a __module__ attribute. For pybind11's
// internal function_record type, we do not have an actual module object to
// attach, so we cannot use PyType_FromModuleAndSpec to set __module__
// automatically.
//
// As a workaround, we define a "qualified" type name that includes a dummy
// module name (PYBIND11_DUMMY_MODULE_NAME). This is non‑idiomatic but avoids
// the deprecation warning, and results in reprs like
//
//     <class 'pybind11_builtins.pybind11_detail_function_record_...'>
//
// even though no real pybind11_builtins module exists. If pybind11 gains an
// actual module object in the future, this code should switch to
// PyType_FromModuleAndSpec and drop the dummy module workaround.
//
// Note that this name is versioned.
#if defined(PYBIND11_OPAQUE_PYOBJECT)
// Opaque wrappers use PEP 697 type data instead of embedding PyObject_HEAD. Name-based
// recognition must keep these representations separate: PyObject_GetTypeData() requires
// a type created with negative basicsize, and the non-opaque reader assumes PyObject_HEAD.
#    define PYBIND11_DETAIL_FUNCTION_RECORD_PYOBJECT_ABI_TAG "_opaque"
#else
#    define PYBIND11_DETAIL_FUNCTION_RECORD_PYOBJECT_ABI_TAG ""
#endif
#define PYBIND11_DETAIL_FUNCTION_RECORD_TP_PLAINNAME                                              \
    "pybind11_detail_function_record_" PYBIND11_DETAIL_FUNCTION_RECORD_ABI_ID                     \
    "_" PYBIND11_PLATFORM_ABI_ID PYBIND11_DETAIL_FUNCTION_RECORD_PYOBJECT_ABI_TAG
constexpr char tp_plainname_impl[] = PYBIND11_DETAIL_FUNCTION_RECORD_TP_PLAINNAME;
constexpr char tp_qualname_impl[]
    = PYBIND11_DUMMY_MODULE_NAME "." PYBIND11_DETAIL_FUNCTION_RECORD_TP_PLAINNAME;
constexpr char tp_module_qualname_impl[]
    = PYBIND11_DUMMY_MODULE_NAME "." PYBIND11_DETAIL_FUNCTION_RECORD_TP_PLAINNAME "_module";

PYBIND11_NAMESPACE_END(function_record_PyTypeObject_methods)

static PyType_Slot function_record_PyType_Slots[] = {
    {Py_tp_dealloc,
     reinterpret_cast<void *>(function_record_PyTypeObject_methods::tp_dealloc_impl)},
    {Py_tp_methods,
     reinterpret_cast<void *>(function_record_PyTypeObject_methods::tp_methods_impl)},
    {Py_tp_init, reinterpret_cast<void *>(function_record_PyTypeObject_methods::tp_init_impl)},
    {Py_tp_alloc, reinterpret_cast<void *>(function_record_PyTypeObject_methods::tp_alloc_impl)},
    {Py_tp_new, reinterpret_cast<void *>(function_record_PyTypeObject_methods::tp_new_impl)},
    {Py_tp_free, reinterpret_cast<void *>(function_record_PyTypeObject_methods::tp_free_impl)},
    {0, nullptr}};

static PyType_Spec function_record_PyType_Spec
    = {function_record_PyTypeObject_methods::tp_qualname_impl,
       PYBIND11_TYPE_DATA_SIZE(function_record_PyObject),
       0,
       Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
       function_record_PyType_Slots};

inline PyTypeObject *get_function_record_PyTypeObject() {
    PYBIND11_LOCK_INTERNALS(get_internals());
    PyTypeObject *&py_type_obj = detail::get_local_internals().function_record_py_type;
    if (!py_type_obj) {
        PyObject *py_obj = PyType_FromSpec(&function_record_PyType_Spec);
        if (py_obj == nullptr) {
            throw error_already_set();
        }
        py_type_obj = reinterpret_cast<PyTypeObject *>(py_obj);
    }
    return py_type_obj;
}

// This works across extension modules. Note that tp_name is versioned.
inline bool function_record_PyTypeObject_name_matches(PyTypeObject *obj_type) {
    return tp_name_equals(obj_type, function_record_PyTypeObject_methods::tp_qualname_impl)
           || tp_name_equals(obj_type, function_record_PyTypeObject_methods::tp_plainname_impl);
}

#if defined(PYBIND11_FUNCTION_RECORD_QUALNAME)
inline PyTypeObject *get_function_record_module_PyTypeObject() {
    PYBIND11_LOCK_INTERNALS(get_internals());
    PyTypeObject *&py_type_obj = detail::get_local_internals().function_record_module_py_type;
    if (!py_type_obj) {
        static PyType_Slot slots[]
            = {{Py_tp_dealloc,
                reinterpret_cast<void *>(function_record_PyTypeObject_methods::tp_dealloc_impl)},
               {Py_tp_methods,
                reinterpret_cast<void *>(function_record_PyTypeObject_methods::tp_methods_impl)},
               // The module `tp_getattro` needs the module `__dict__`.
               {Py_tp_getattro, reinterpret_cast<void *>(PyObject_GenericGetAttr)},
               {0, nullptr}};
        PyType_Spec spec = {function_record_PyTypeObject_methods::tp_module_qualname_impl,
#    if PY_VERSION_HEX >= 0x030C0000
                            -static_cast<int>(sizeof(function_record *)),
#    else
                            static_cast<int>(PyModule_Type.tp_basicsize
                                             + static_cast<Py_ssize_t>(sizeof(function_record *))),
#    endif
                            0,
                            Py_TPFLAGS_DEFAULT,
                            slots};
        PyObject *py_obj
            = PyType_FromSpecWithBases(&spec, reinterpret_cast<PyObject *>(&PyModule_Type));
        if (py_obj == nullptr) {
            throw error_already_set();
        }
        py_type_obj = reinterpret_cast<PyTypeObject *>(py_obj);
    }
    return py_type_obj;
}

inline bool is_function_record_module_PyObject(PyObject *obj) {
    PyTypeObject *obj_type = Py_TYPE(obj);
    return obj_type == get_function_record_module_PyTypeObject()
           || tp_name_equals(obj_type,
                             function_record_PyTypeObject_methods::tp_module_qualname_impl);
}

// The function record type for the functions in the class `scope`. It is in the class
// `__dict__`, so that all extension modules use the same one.
inline object get_function_record_PyTypeObject(handle scope) {
    constexpr const char *key = "__pybind11_function_record_type__";
    object scope_dict = scope.attr("__dict__");
    if (scope_dict.contains(key)) {
        return scope_dict[key];
    }
    auto py_type_obj = reinterpret_steal<object>(PyType_FromSpec(&function_record_PyType_Spec));
    if (!py_type_obj) {
        throw error_already_set();
    }
    setattr(py_type_obj, "__qualname__", scope.attr("__qualname__"));
    setattr(scope, key, py_type_obj);
    return py_type_obj;
}
#endif

inline bool is_function_record_PyObject(PyObject *obj) {
    if (PyType_Check(obj) != 0) {
        return false;
    }
    PyTypeObject *obj_type = Py_TYPE(obj);

    PyTypeObject *frtype = get_function_record_PyTypeObject();

    // Fast path (pointer comparison).
    if (obj_type == frtype) {
        return true;
    }
    return function_record_PyTypeObject_name_matches(obj_type);
}

/// The location of the record pointer in a function record object of any layout, or nullptr.
inline function_record **function_record_ptr_location(PyObject *obj) {
#if defined(PYBIND11_FUNCTION_RECORD_QUALNAME)
    if (is_function_record_module_PyObject(obj)) {
        return &function_record_module_data(obj);
    }
#endif
    if (is_function_record_PyObject(obj)) {
        return &function_record_data(obj)->cpp_func_rec;
    }
    return nullptr;
}

inline function_record *function_record_ptr_from_PyObject(PyObject *obj) {
    function_record **location = function_record_ptr_location(obj);
    return location != nullptr ? *location : nullptr;
}

// The `self` of cpp_function::dispatcher() is always the function_record_PyObject that was
// created for that dispatcher in cpp_function::initialize_generic(), so the type check can be
// skipped. is_function_record_PyObject() is deliberately not used here: it calls
// get_function_record_PyTypeObject(), which acquires the internals lock on every call.
inline function_record *function_record_ptr_from_dispatcher_self(PyObject *self) {
    assert(self != nullptr);
    assert(PyType_Check(self) == 0);
    assert(function_record_PyTypeObject_name_matches(Py_TYPE(self)));
    return function_record_data(self)->cpp_func_rec;
}

#if defined(PYBIND11_FUNCTION_RECORD_QUALNAME)
// Same as function_record_ptr_from_dispatcher_self(), for a record that is a module subclass.
inline function_record *function_record_ptr_from_dispatcher_module_self(PyObject *self) {
    assert(self != nullptr);
    assert(PyModule_Check(self) != 0);
    return function_record_module_data(self);
}
#endif

/// A new function record object for a function in `scope`, which holds `rec`.
inline object function_record_PyObject_New(handle scope, function_record *rec) {
#if defined(PYBIND11_FUNCTION_RECORD_QUALNAME)
    if (!scope || PyType_Check(scope.ptr()) == 0) {
        // Not the module `tp_new`: it adds a `__dict__`, and the free-threaded build then
        // defers the deallocation to the garbage collector.
        auto py_func_rec = reinterpret_steal<object>(
            PyType_GenericAlloc(get_function_record_module_PyTypeObject(), 0));
        if (!py_func_rec) {
            throw error_already_set();
        }
        function_record_module_data(py_func_rec.ptr()) = rec;
        return py_func_rec;
    }
    object py_type_obj = get_function_record_PyTypeObject(scope);
    PyObject *py_func_rec = PyObject_New(PyObject, (PyTypeObject *) py_type_obj.ptr());
#else
    (void) scope;
    PyObject *py_func_rec = PyObject_New(PyObject, get_function_record_PyTypeObject());
#endif
    if (py_func_rec == nullptr) {
        throw error_already_set();
    }
    function_record_data(py_func_rec)->cpp_func_rec = rec;
    return reinterpret_steal<object>(py_func_rec);
}

PYBIND11_NAMESPACE_BEGIN(function_record_PyTypeObject_methods)

// Guard against accidents & oversights, in particular when porting to future Python versions.
inline PyObject *tp_new_impl(PyTypeObject *, PyObject *, PyObject *) {
    pybind11_fail("UNEXPECTED CALL OF function_record_PyTypeObject_methods::tp_new_impl");
    // return nullptr; // Unreachable.
}

inline PyObject *tp_alloc_impl(PyTypeObject *, Py_ssize_t) {
    pybind11_fail("UNEXPECTED CALL OF function_record_PyTypeObject_methods::tp_alloc_impl");
    // return nullptr; // Unreachable.
}

inline int tp_init_impl(PyObject *, PyObject *, PyObject *) {
    pybind11_fail("UNEXPECTED CALL OF function_record_PyTypeObject_methods::tp_init_impl");
    // return -1; // Unreachable.
}

inline void tp_free_impl(void *) {
    pybind11_fail("UNEXPECTED CALL OF function_record_PyTypeObject_methods::tp_free_impl");
}

inline PyObject *reduce_ex_impl(PyObject *self, PyObject *, PyObject *) {
    // Deliberately ignoring the arguments for simplicity (expected is `protocol: int`).
    const function_record *rec = function_record_ptr_from_PyObject(self);
    if (rec == nullptr) {
        pybind11_fail(
            "FATAL: function_record_PyTypeObject reduce_ex_impl(): cannot obtain cpp_func_rec.");
    }
    if (rec->name != nullptr && rec->name[0] != '\0' && rec->scope
        && PyModule_Check(rec->scope.ptr()) != 0) {
        object scope_module = get_scope_module(rec->scope);
        if (scope_module) {
            auto builtins = reinterpret_borrow<dict>(PyEval_GetBuiltins());
            auto builtins_eval = builtins["eval"];
            auto reconstruct_args = make_tuple(str("__import__('importlib').import_module('")
                                               + scope_module + str("')"));
            return make_tuple(std::move(builtins_eval), std::move(reconstruct_args))
                .release()
                .ptr();
        }
    }
    set_error(PyExc_RuntimeError, repr(self) + str(" is not pickleable."));
    return nullptr;
}

PYBIND11_NAMESPACE_END(function_record_PyTypeObject_methods)

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
