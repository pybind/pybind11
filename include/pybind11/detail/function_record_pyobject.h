// Copyright (c) 2024-2025 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

// For background see the description of PR google/pybind11clif#30099.

#pragma once

#include <pybind11/attr.h>
#include <pybind11/conduit/pybind11_platform_abi_id.h>
#include <pybind11/pytypes.h>

#include "common.h"

#include <cstring>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

struct function_record_PyObject {
    PyObject_HEAD
    function_record *cpp_func_rec;
};

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

// Note that this name is versioned.
constexpr char tp_name_impl[]
    = "pybind11_detail_function_record_" PYBIND11_DETAIL_FUNCTION_RECORD_ABI_ID
      "_" PYBIND11_PLATFORM_ABI_ID;

PYBIND11_NAMESPACE_END(function_record_PyTypeObject_methods)

// Designated initializers are a C++20 feature:
// https://en.cppreference.com/w/cpp/language/aggregate_initialization#Designated_initializers
// MSVC rejects them unless /std:c++20 is used (error code C7555).
PYBIND11_WARNING_PUSH
PYBIND11_WARNING_DISABLE_CLANG("-Wmissing-field-initializers")
#if defined(__GNUC__) && __GNUC__ >= 8
PYBIND11_WARNING_DISABLE_GCC("-Wmissing-field-initializers")
#endif
static PyTypeObject function_record_PyTypeObject = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    /* const char *tp_name */ function_record_PyTypeObject_methods::tp_name_impl,
    /* Py_ssize_t tp_basicsize */ sizeof(function_record_PyObject),
    /* Py_ssize_t tp_itemsize */ 0,
    /* destructor tp_dealloc */ function_record_PyTypeObject_methods::tp_dealloc_impl,
    /* Py_ssize_t tp_vectorcall_offset */ 0,
    /* getattrfunc tp_getattr */ nullptr,
    /* setattrfunc tp_setattr */ nullptr,
    /* PyAsyncMethods *tp_as_async */ nullptr,
    /* reprfunc tp_repr */ nullptr,
    /* PyNumberMethods *tp_as_number */ nullptr,
    /* PySequenceMethods *tp_as_sequence */ nullptr,
    /* PyMappingMethods *tp_as_mapping */ nullptr,
    /* hashfunc tp_hash */ nullptr,
    /* ternaryfunc tp_call */ nullptr,
    /* reprfunc tp_str */ nullptr,
    /* getattrofunc tp_getattro */ nullptr,
    /* setattrofunc tp_setattro */ nullptr,
    /* PyBufferProcs *tp_as_buffer */ nullptr,
    /* unsigned long tp_flags */ Py_TPFLAGS_DEFAULT,
    /* const char *tp_doc */ nullptr,
    /* traverseproc tp_traverse */ nullptr,
    /* inquiry tp_clear */ nullptr,
    /* richcmpfunc tp_richcompare */ nullptr,
    /* Py_ssize_t tp_weaklistoffset */ 0,
    /* getiterfunc tp_iter */ nullptr,
    /* iternextfunc tp_iternext */ nullptr,
    /* struct PyMethodDef *tp_methods */ function_record_PyTypeObject_methods::tp_methods_impl,
    /* struct PyMemberDef *tp_members */ nullptr,
    /* struct PyGetSetDef *tp_getset */ nullptr,
    /* struct _typeobject *tp_base */ nullptr,
    /* PyObject *tp_dict */ nullptr,
    /* descrgetfunc tp_descr_get */ nullptr,
    /* descrsetfunc tp_descr_set */ nullptr,
    /* Py_ssize_t tp_dictoffset */ 0,
    /* initproc tp_init */ function_record_PyTypeObject_methods::tp_init_impl,
    /* allocfunc tp_alloc */ function_record_PyTypeObject_methods::tp_alloc_impl,
    /* newfunc tp_new */ function_record_PyTypeObject_methods::tp_new_impl,
    /* freefunc tp_free */ function_record_PyTypeObject_methods::tp_free_impl,
    /* inquiry tp_is_gc */ nullptr,
    /* PyObject *tp_bases */ nullptr,
    /* PyObject *tp_mro */ nullptr,
    /* PyObject *tp_cache */ nullptr,
    /* PyObject *tp_subclasses */ nullptr,
    /* PyObject *tp_weaklist */ nullptr,
    /* destructor tp_del */ nullptr,
    /* unsigned int tp_version_tag */ 0,
    /* destructor tp_finalize */ nullptr,
    /* vectorcallfunc tp_vectorcall */ nullptr,
};
PYBIND11_WARNING_POP

static bool function_record_PyTypeObject_PyType_Ready_first_call = true;

inline void function_record_PyTypeObject_PyType_Ready() {
    if (function_record_PyTypeObject_PyType_Ready_first_call) {
        if (PyType_Ready(&function_record_PyTypeObject) < 0) {
            throw error_already_set();
        }
        function_record_PyTypeObject_PyType_Ready_first_call = false;
    }
}

inline bool is_function_record_PyObject(PyObject *obj) {
    if (PyType_Check(obj) != 0) {
        return false;
    }
    PyTypeObject *obj_type = Py_TYPE(obj);
    // Fast path (pointer comparison).
    if (obj_type == &function_record_PyTypeObject) {
        return true;
    }
    // This works across extension modules. Note that tp_name is versioned.
    if (strcmp(obj_type->tp_name, function_record_PyTypeObject.tp_name) == 0) {
        return true;
    }
    return false;
}

inline function_record *function_record_ptr_from_PyObject(PyObject *obj) {
    if (is_function_record_PyObject(obj)) {
        return ((detail::function_record_PyObject *) obj)->cpp_func_rec;
    }
    return nullptr;
}

inline object function_record_PyObject_New() {
    auto *py_func_rec = PyObject_New(function_record_PyObject, &function_record_PyTypeObject);
    if (py_func_rec == nullptr) {
        throw error_already_set();
    }
    py_func_rec->cpp_func_rec = nullptr; // For clarity/purity. Redundant in practice.
    return reinterpret_steal<object>((PyObject *) py_func_rec);
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
            return make_tuple(reinterpret_borrow<object>(PyEval_GetBuiltins())["eval"],
                              make_tuple(str("__import__('importlib').import_module('")
                                         + scope_module + str("')")))
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
