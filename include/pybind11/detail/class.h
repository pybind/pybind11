/*
    pybind11/detail/class.h: Python C API implementation details for py::class_

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include <pybind11/attr.h>
#include <pybind11/options.h>

#include "exception_translation.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

#if !defined(PYPY_VERSION)
#    define PYBIND11_BUILTIN_QUALNAME
#    define PYBIND11_SET_OLDPY_QUALNAME(obj, nameobj)
#else
// In PyPy, we still set __qualname__ so that we can produce reliable function type
// signatures; in CPython this macro expands to nothing:
#    define PYBIND11_SET_OLDPY_QUALNAME(obj, nameobj)                                             \
        setattr((PyObject *) obj, "__qualname__", nameobj)
#endif

std::string get_fully_qualified_tp_name(PyTypeObject *type);

PyTypeObject *type_incref(PyTypeObject *type);

#if !defined(PYPY_VERSION)

/// `pybind11_static_property.__get__()`: Always pass the class instead of the instance.
extern "C" PyObject *pybind11_static_get(PyObject *self, PyObject * /*ob*/, PyObject *cls);

/// `pybind11_static_property.__set__()`: Just like the above `__get__()`.
extern "C" int pybind11_static_set(PyObject *self, PyObject *obj, PyObject *value);

/** A `static_property` is the same as a `property` but the `__get__()` and `__set__()`
    methods are modified to always use the object type instead of a concrete instance.
    Return value: New reference. */
PyTypeObject *make_static_property_type();

#else // PYPY

/** PyPy has some issues with the above C API, so we evaluate Python code instead.
    This function will only be called once so performance isn't really a concern.
    Return value: New reference. */
PyTypeObject *make_static_property_type();

#endif // PYPY

/** Types with static properties need to handle `Type.static_prop = x` in a specific way.
    By default, Python replaces the `static_property` itself, but for wrapped C++ types
    we need to call `static_property.__set__()` in order to propagate the new value to
    the underlying C++ data structure. */
extern "C" int pybind11_meta_setattro(PyObject *obj, PyObject *name, PyObject *value);

/**
 * Python 3's PyInstanceMethod_Type hides itself via its tp_descr_get, which prevents aliasing
 * methods via cls.attr("m2") = cls.attr("m1"): instead the tp_descr_get returns a plain function,
 * when called on a class, or a PyMethod, when called on an instance.  Override that behaviour here
 * to do a special case bypass for PyInstanceMethod_Types.
 */
extern "C" PyObject *pybind11_meta_getattro(PyObject *obj, PyObject *name);

/// metaclass `__call__` function that is used to create all pybind11 objects.
extern "C" PyObject *pybind11_meta_call(PyObject *type, PyObject *args, PyObject *kwargs);

/// Cleanup the type-info for a pybind11-registered type.
extern "C" void pybind11_meta_dealloc(PyObject *obj);

/** This metaclass is assigned by default to all pybind11 types and is required in order
    for static properties to function correctly. Users may override this using `py::metaclass`.
    Return value: New reference. */
PyTypeObject *make_default_metaclass();

/// For multiple inheritance types we need to recursively register/deregister base pointers for any
/// base classes with pointers that are difference from the instance value pointer so that we can
/// correctly recognize an offset base class pointer. This calls a function with any offset base
/// ptrs.
void traverse_offset_bases(void *valueptr,
                           const detail::type_info *tinfo,
                           instance *self,
                           bool (*f)(void * /*parentptr*/, instance * /*self*/));

#ifdef Py_GIL_DISABLED
void enable_try_inc_ref(PyObject *obj);
#endif

bool register_instance_impl(void *ptr, instance *self);
bool deregister_instance_impl(void *ptr, instance *self);

void register_instance(instance *self, void *valptr, const type_info *tinfo);

bool deregister_instance(instance *self, void *valptr, const type_info *tinfo);

/// Instance creation function for all pybind11 types. It allocates the internal instance layout
/// for holding C++ objects and holders.  Allocation is done lazily (the first time the instance is
/// cast to a reference or pointer), and initialization is done by an `__init__` function.
PyObject *make_new_instance(PyTypeObject *type);

/// Instance creation function for all pybind11 types. It only allocates space for the
/// C++ object, but doesn't call the constructor -- an `__init__` function must do that.
extern "C" PyObject *pybind11_object_new(PyTypeObject *type, PyObject *, PyObject *);

/// An `__init__` function constructs the C++ object. Users should provide at least one
/// of these using `py::init` or directly with `.def(__init__, ...)`. Otherwise, the
/// following default function will be used which simply throws an exception.
extern "C" int pybind11_object_init(PyObject *self, PyObject *, PyObject *);

void add_patient(PyObject *nurse, PyObject *patient);

void clear_patients(PyObject *self);

/// Clears all internal data from the instance and removes it from registered instances in
/// preparation for deallocation.
void clear_instance(PyObject *self);

/// Instance destructor function for all pybind11 types. It calls `type_info.dealloc`
/// to destroy the C++ object itself, while the rest is Python bookkeeping.
extern "C" void pybind11_object_dealloc(PyObject *self);

PYBIND11_WARNING_PUSH
PYBIND11_WARNING_DISABLE_GCC("-Wredundant-decls")

std::string error_string();

PYBIND11_WARNING_POP

/** Create the type which can be used as a common base for all classes.  This is
    needed in order to satisfy Python's requirements for multiple inheritance.
    Return value: New reference. */
PyObject *make_object_base_type(PyTypeObject *metaclass);

/// dynamic_attr: Allow the garbage collector to traverse the internal instance `__dict__`.
extern "C" int pybind11_traverse(PyObject *self, visitproc visit, void *arg);

/// dynamic_attr: Allow the GC to clear the dictionary.
extern "C" int pybind11_clear(PyObject *self);

/// Give instances of this type a `__dict__` and opt into garbage collection.
void enable_dynamic_attributes(PyHeapTypeObject *heap_type);

/// buffer_protocol: Fill in the view as specified by flags.
extern "C" int pybind11_getbuffer(PyObject *obj, Py_buffer *view, int flags);

/// buffer_protocol: Release the resources of the buffer.
extern "C" void pybind11_releasebuffer(PyObject *, Py_buffer *view);

/// Give this type a buffer interface.
void enable_buffer_protocol(PyHeapTypeObject *heap_type);

/** Create a brand new Python type according to the `type_record` specification.
    Return value: New reference. */
PyObject *make_new_python_type(const type_record &rec);

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

#ifndef PYBIND11_PRECOMPILED
#    include "class-inl.h" // IWYU pragma: export
#endif
