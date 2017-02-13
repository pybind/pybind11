/*
    pybind11/class_support.h: Python C API implementation details for py::class_

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "attr.h"

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

#if !defined(PYPY_VERSION)

/// `pybind11_static_property.__get__()`: Always pass the class instead of the instance.
extern "C" inline PyObject *pybind11_static_get(PyObject *self, PyObject * /*ob*/, PyObject *cls) {
    return PyProperty_Type.tp_descr_get(self, cls, cls);
}

/// `pybind11_static_property.__set__()`: Just like the above `__get__()`.
extern "C" inline int pybind11_static_set(PyObject *self, PyObject *obj, PyObject *value) {
    PyObject *cls = PyType_Check(obj) ? obj : (PyObject *) Py_TYPE(obj);
    return PyProperty_Type.tp_descr_set(self, cls, value);
}

/** A `static_property` is the same as a `property` but the `__get__()` and `__set__()`
    methods are modified to always use the object type instead of a concrete instance.
    Return value: New reference. */
inline PyTypeObject *make_static_property_type() {
    constexpr auto *name = "pybind11_static_property";
    auto name_obj = reinterpret_steal<object>(PYBIND11_FROM_STRING(name));

    /* Danger zone: from now (and until PyType_Ready), make sure to
       issue no Python C API calls which could potentially invoke the
       garbage collector (the GC will call type_traverse(), which will in
       turn find the newly constructed type in an invalid state) */
    auto heap_type = (PyHeapTypeObject *) PyType_Type.tp_alloc(&PyType_Type, 0);
    if (!heap_type)
        pybind11_fail("make_static_property_type(): error allocating type!");

    heap_type->ht_name = name_obj.inc_ref().ptr();
#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 3
    heap_type->ht_qualname = name_obj.inc_ref().ptr();
#endif

    auto type = &heap_type->ht_type;
    type->tp_name = name;
    type->tp_base = &PyProperty_Type;
    type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;
    type->tp_descr_get = pybind11_static_get;
    type->tp_descr_set = pybind11_static_set;

    if (PyType_Ready(type) < 0)
        pybind11_fail("make_static_property_type(): failure in PyType_Ready()!");

    return type;
}

#else // PYPY

/** PyPy has some issues with the above C API, so we evaluate Python code instead.
    This function will only be called once so performance isn't really a concern.
    Return value: New reference. */
inline PyTypeObject *make_static_property_type() {
    auto d = dict();
    PyObject *result = PyRun_String(R"(\
        class pybind11_static_property(property):
            def __get__(self, obj, cls):
                return property.__get__(self, cls, cls)

            def __set__(self, obj, value):
                cls = obj if isinstance(obj, type) else type(obj)
                property.__set__(self, cls, value)
        )", Py_file_input, d.ptr(), d.ptr()
    );
    if (result == nullptr)
        throw error_already_set();
    Py_DECREF(result);
    return (PyTypeObject *) d["pybind11_static_property"].cast<object>().release().ptr();
}

#endif // PYPY

/** Types with static properties need to handle `Type.static_prop = x` in a specific way.
    By default, Python replaces the `static_property` itself, but for wrapped C++ types
    we need to call `static_property.__set__()` in order to propagate the new value to
    the underlying C++ data structure. */
extern "C" inline int pybind11_meta_setattro(PyObject* obj, PyObject* name, PyObject* value) {
    // Use `_PyType_Lookup()` instead of `PyObject_GetAttr()` in order to get the raw
    // descriptor (`property`) instead of calling `tp_descr_get` (`property.__get__()`).
    PyObject *descr = _PyType_Lookup((PyTypeObject *) obj, name);

    // Call `static_property.__set__()` instead of replacing the `static_property`.
    if (descr && PyObject_IsInstance(descr, (PyObject *) get_internals().static_property_type)) {
#if !defined(PYPY_VERSION)
        return Py_TYPE(descr)->tp_descr_set(descr, obj, value);
#else
        if (PyObject *result = PyObject_CallMethod(descr, "__set__", "OO", obj, value)) {
            Py_DECREF(result);
            return 0;
        } else {
            return -1;
        }
#endif
    } else {
        return PyType_Type.tp_setattro(obj, name, value);
    }
}

/** This metaclass is assigned by default to all pybind11 types and is required in order
    for static properties to function correctly. Users may override this using `py::metaclass`.
    Return value: New reference. */
inline PyTypeObject* make_default_metaclass() {
    constexpr auto *name = "pybind11_type";
    auto name_obj = reinterpret_steal<object>(PYBIND11_FROM_STRING(name));

    /* Danger zone: from now (and until PyType_Ready), make sure to
       issue no Python C API calls which could potentially invoke the
       garbage collector (the GC will call type_traverse(), which will in
       turn find the newly constructed type in an invalid state) */
    auto heap_type = (PyHeapTypeObject *) PyType_Type.tp_alloc(&PyType_Type, 0);
    if (!heap_type)
        pybind11_fail("make_default_metaclass(): error allocating metaclass!");

    heap_type->ht_name = name_obj.inc_ref().ptr();
#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 3
    heap_type->ht_qualname = name_obj.inc_ref().ptr();
#endif

    auto type = &heap_type->ht_type;
    type->tp_name = name;
    type->tp_base = &PyType_Type;
    type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;
    type->tp_setattro = pybind11_meta_setattro;

    if (PyType_Ready(type) < 0)
        pybind11_fail("make_default_metaclass(): failure in PyType_Ready()!");

    return type;
}

NAMESPACE_END(detail)
NAMESPACE_END(pybind11)
