/*
    pybind11/detail/class-inl.h: Out-of-line definitions for class.h

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

// Every function defined here must start with PYBIND11_INLINE (or
// PYBIND11_NOINLINE_ATTR PYBIND11_INLINE). In the default header-only mode this file is
// included at the bottom of class.h; when PYBIND11_PRECOMPILED is defined it is only
// compiled into the pybind11 static library (see src/).

#pragma once

#include "class.h"

#include <algorithm>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

PYBIND11_INLINE std::string get_fully_qualified_tp_name(PyTypeObject *type) {
#if !defined(PYPY_VERSION)
    return get_tp_name(type);
#else
    auto module_name = handle((PyObject *) type).attr("__module__").cast<std::string>();
    if (module_name == PYBIND11_BUILTINS_MODULE)
        return get_tp_name(type);
    else
        return std::move(module_name) + "." + get_tp_name(type);
#endif
}

PYBIND11_INLINE PyTypeObject *type_incref(PyTypeObject *type) {
    Py_INCREF(reinterpret_cast<PyObject *>(type));
    return type;
}

// Slots of the static base types that the pybind11 metaclass and static property type forward
// to. PyType_GetSlot() works on static types since Python 3.10 and is part of the stable ABI;
// the function-local caches make each lookup happen once per module. Below 3.10 (and on PyPy,
// where PyType_GetSlot on static types is not reliable) the struct fields are read directly.
#if PY_VERSION_HEX >= 0x030A0000 && !defined(PYPY_VERSION)
#    define PYBIND11_BASE_TYPE_SLOT(type, slot_id, field, slot_type)                              \
        static const auto cached = reinterpret_cast<slot_type>(PyType_GetSlot(&type, slot_id));   \
        return cached;
#else
#    define PYBIND11_BASE_TYPE_SLOT(type, slot_id, field, slot_type) return type.field;
#endif

PYBIND11_INLINE ternaryfunc type_type_call(){PYBIND11_BASE_TYPE_SLOT(
    PyType_Type, Py_tp_call, tp_call, ternaryfunc)} PYBIND11_INLINE setattrofunc
    type_type_setattro(){PYBIND11_BASE_TYPE_SLOT(
        PyType_Type, Py_tp_setattro, tp_setattro, setattrofunc)} PYBIND11_INLINE getattrofunc
    type_type_getattro(){PYBIND11_BASE_TYPE_SLOT(
        PyType_Type, Py_tp_getattro, tp_getattro, getattrofunc)} PYBIND11_INLINE destructor
    type_type_dealloc(){PYBIND11_BASE_TYPE_SLOT(
        PyType_Type, Py_tp_dealloc, tp_dealloc, destructor)} PYBIND11_INLINE descrgetfunc
    property_type_descr_get(){PYBIND11_BASE_TYPE_SLOT(
        PyProperty_Type, Py_tp_descr_get, tp_descr_get, descrgetfunc)} PYBIND11_INLINE descrsetfunc
    property_type_descr_set() {
    PYBIND11_BASE_TYPE_SLOT(PyProperty_Type, Py_tp_descr_set, tp_descr_set, descrsetfunc)
}

#undef PYBIND11_BASE_TYPE_SLOT

#if !defined(PYPY_VERSION)
extern "C" PYBIND11_INLINE PyObject *
pybind11_static_get(PyObject *self, PyObject * /*ob*/, PyObject *cls) {
    return property_type_descr_get()(self, cls, cls);
}

extern "C" PYBIND11_INLINE int
pybind11_static_set(PyObject *self, PyObject *obj, PyObject *value) {
    PyObject *cls = PyType_Check(obj) ? obj : (PyObject *) Py_TYPE(obj);
    return property_type_descr_set()(self, cls, value);
}

#    if defined(PYBIND11_TYPE_CREATION_VIA_SPEC)

#        if defined(Py_LIMITED_API)
// Without Py_TPFLAGS_MANAGED_DICT the `__dict__` slot follows the property object. These two
// functions find it through the type's `__dictoffset__`.
PYBIND11_INLINE PyObject **static_property_dict_ptr(PyObject *self) {
    static const Py_ssize_t offset = handle(reinterpret_cast<PyObject *>(Py_TYPE(self)))
                                         .attr("__dictoffset__")
                                         .cast<Py_ssize_t>();
    return reinterpret_cast<PyObject **>(reinterpret_cast<char *>(self) + offset);
}
extern "C" PYBIND11_INLINE int
pybind11_static_property_traverse(PyObject *self, visitproc visit, void *arg) {
    Py_VISIT(*static_property_dict_ptr(self));
    Py_VISIT(Py_TYPE(self));
    return 0;
}
extern "C" PYBIND11_INLINE int pybind11_static_property_clear(PyObject *self) {
    Py_CLEAR(*static_property_dict_ptr(self));
    return 0;
}
#        endif

PYBIND11_INLINE PyTypeObject *make_static_property_type() {
    // Since Python-3.12 property-derived types are required to have dynamic attributes (to set
    // `__doc__`), hence the GC and dict slots.
#        if !defined(Py_LIMITED_API)
    static PyType_Slot slots[] = {{Py_tp_descr_get, reinterpret_cast<void *>(pybind11_static_get)},
                                  {Py_tp_descr_set, reinterpret_cast<void *>(pybind11_static_set)},
                                  {Py_tp_traverse, reinterpret_cast<void *>(pybind11_traverse)},
                                  {Py_tp_clear, reinterpret_cast<void *>(pybind11_clear)},
                                  {Py_tp_getset, reinterpret_cast<void *>(dynamic_attr_getset())},
                                  {0, nullptr}};
    static PyType_Spec spec
        = {PYBIND11_DUMMY_MODULE_NAME ".pybind11_static_property",
           0, // inherit from property
           0,
           Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_MANAGED_DICT,
           slots};
#        else
    // Append the `__dict__` slot to the property layout. (PEP 697 relative offsets are not
    // applied to `__dictoffset__` on 3.12, so the offset is absolute.)
    const auto dictoffset = type_generic_getattr(&PyProperty_Type, "__basicsize__").cast<int>();
    PyMemberDef members[] = {{"__dictoffset__", Py_T_PYSSIZET, dictoffset, Py_READONLY, nullptr},
                             {nullptr, 0, 0, 0, nullptr}};
    PyType_Slot slots[]
        = {{Py_tp_descr_get, reinterpret_cast<void *>(pybind11_static_get)},
           {Py_tp_descr_set, reinterpret_cast<void *>(pybind11_static_set)},
           {Py_tp_traverse, reinterpret_cast<void *>(pybind11_static_property_traverse)},
           {Py_tp_clear, reinterpret_cast<void *>(pybind11_static_property_clear)},
           {Py_tp_getset, reinterpret_cast<void *>(dynamic_attr_getset())},
           {Py_tp_members, reinterpret_cast<void *>(members)},
           {0, nullptr}};
    PyType_Spec spec = {PYBIND11_DUMMY_MODULE_NAME ".pybind11_static_property",
                        dictoffset + static_cast<int>(sizeof(PyObject *)),
                        0,
                        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
                        slots};
#        endif
    PyObject *type = PyType_FromMetaclass(
        nullptr, nullptr, &spec, reinterpret_cast<PyObject *>(&PyProperty_Type));
    if (!type) {
        pybind11_fail("make_static_property_type(): failure in PyType_FromMetaclass(): "
                      + error_string());
    }
    return reinterpret_cast<PyTypeObject *>(type);
}

#    else // legacy: fill in a PyHeapTypeObject by hand

PYBIND11_INLINE PyTypeObject *make_static_property_type() {
    constexpr auto *name = "pybind11_static_property";
    auto name_obj = reinterpret_steal<object>(PYBIND11_FROM_STRING(name));

    /* Danger zone: from now (and until PyType_Ready), make sure to
       issue no Python C API calls which could potentially invoke the
       garbage collector (the GC will call type_traverse(), which will in
       turn find the newly constructed type in an invalid state) */
    auto *heap_type = reinterpret_cast<PyHeapTypeObject *>(PyType_Type.tp_alloc(&PyType_Type, 0));
    if (!heap_type) {
        pybind11_fail("make_static_property_type(): error allocating type!");
    }

    heap_type->ht_name = name_obj.inc_ref().ptr();
#        ifdef PYBIND11_BUILTIN_QUALNAME
    heap_type->ht_qualname = name_obj.inc_ref().ptr();
#        endif

    auto *type = &heap_type->ht_type;
    type->tp_name = name;
    type->tp_base = type_incref(&PyProperty_Type);
    type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;
    type->tp_descr_get = pybind11_static_get;
    type->tp_descr_set = pybind11_static_set;

#        if PY_VERSION_HEX >= 0x030C0000
    // Since Python-3.12 property-derived types are required to
    // have dynamic attributes (to set `__doc__`)
    enable_dynamic_attributes(heap_type);
#        endif

    if (PyType_Ready(type) < 0) {
        pybind11_fail("make_static_property_type(): failure in PyType_Ready()!");
    }

    setattr(reinterpret_cast<PyObject *>(type), "__module__", str(PYBIND11_DUMMY_MODULE_NAME));
    PYBIND11_SET_OLDPY_QUALNAME(type, name_obj);

    return type;
}

#    endif // PYBIND11_TYPE_CREATION_VIA_SPEC

#else // PYPY
PYBIND11_INLINE PyTypeObject *make_static_property_type() {
    auto d = dict();
    PyObject *result = PyRun_String(R"(\
class pybind11_static_property(property):
    def __get__(self, obj, cls):
        return property.__get__(self, cls, cls)

    def __set__(self, obj, value):
        cls = obj if isinstance(obj, type) else type(obj)
        property.__set__(self, cls, value)
)",
                                    Py_file_input,
                                    d.ptr(),
                                    d.ptr());
    if (result == nullptr)
        throw error_already_set();
    Py_DECREF(result);
    return (PyTypeObject *) d["pybind11_static_property"].cast<object>().release().ptr();
}

#endif // PYPY
extern "C" PYBIND11_INLINE int
pybind11_meta_setattro(PyObject *obj, PyObject *name, PyObject *value) {
    // Use `type_lookup()` instead of `PyObject_GetAttr()` in order to get the raw
    // descriptor (`property`) instead of calling `tp_descr_get` (`property.__get__()`).
    object descr = type_lookup((PyTypeObject *) obj, name);

    // The following assignment combinations are possible:
    //   1. `Type.static_prop = value`             --> descr_set: `Type.static_prop.__set__(value)`
    //   2. `Type.static_prop = other_static_prop` --> setattro:  replace existing `static_prop`
    //   3. `Type.regular_attribute = value`       --> setattro:  regular attribute assignment
    auto *const static_prop = (PyObject *) get_internals().static_property_type;
    const auto call_descr_set = descr && (value != nullptr)
                                && (PyObject_IsInstance(descr.ptr(), static_prop) != 0)
                                && (PyObject_IsInstance(value, static_prop) == 0);
    if (call_descr_set) {
        // Call `static_property.__set__()` instead of replacing the `static_property`.
#if defined(PYBIND11_HAS_DIRECT_STRUCT_ACCESS)
        return Py_TYPE(descr.ptr())->tp_descr_set(descr.ptr(), obj, value);
#else
        if (PyObject *result = PyObject_CallMethod(descr.ptr(), "__set__", "OO", obj, value)) {
            Py_DECREF(result);
            return 0;
        } else {
            return -1;
        }
#endif
    } else {
        // Replace existing attribute.
        return type_type_setattro()(obj, name, value);
    }
}

extern "C" PYBIND11_INLINE PyObject *pybind11_meta_getattro(PyObject *obj, PyObject *name) {
    object descr = type_lookup((PyTypeObject *) obj, name);
    if (descr && PYBIND11_INSTANCE_METHOD_CHECK(descr.ptr())) {
        return descr.release().ptr();
    }
    return type_type_getattro()(obj, name);
}

extern "C" PYBIND11_INLINE PyObject *
pybind11_meta_call(PyObject *type, PyObject *args, PyObject *kwargs) {

    // use the default metaclass call to create/initialize the object
    PyObject *self = type_type_call()(type, args, kwargs);
    if (self == nullptr) {
        return nullptr;
    }

    // Ensure that the base __init__ function(s) were called
    values_and_holders vhs(self);
    for (const auto &vh : vhs) {
        if (!vh.holder_constructed() && !vhs.is_redundant_value_and_holder(vh)) {
            PyErr_Format(PyExc_TypeError,
                         "%.200s.__init__() must be called when overriding __init__",
                         get_fully_qualified_tp_name(vh.type->type).c_str());
            Py_DECREF(self);
            return nullptr;
        }
    }

    return self;
}

extern "C" PYBIND11_INLINE void pybind11_meta_dealloc(PyObject *obj) {
    with_internals_if_internals([obj](internals &internals) {
        auto *type = (PyTypeObject *) obj;

        // A pybind11-registered type will:
        // 1) be found in internals.registered_types_py
        // 2) have exactly one associated `detail::type_info`
        auto found_type = internals.registered_types_py.find(type);
        if (found_type != internals.registered_types_py.end() && found_type->second.size() == 1
            && found_type->second[0]->type == type) {

            auto *tinfo = found_type->second[0];
            auto tindex = std::type_index(*tinfo->cpptype);
            internals.direct_conversions.erase(tindex);

            auto &local_internals = get_local_internals();
            if (tinfo->module_local) {
                local_internals.registered_types_cpp.erase(tinfo->cpptype);
            } else {
                internals.registered_types_cpp.erase(tindex);
#if PYBIND11_INTERNALS_VERSION >= 12
                internals.registered_types_cpp_fast.erase(tinfo->cpptype);
                for (const std::type_info *alias : tinfo->alias_chain) {
                    auto num_erased = internals.registered_types_cpp_fast.erase(alias);
                    (void) num_erased;
                    assert(num_erased > 0);
                }
#endif
            }
            internals.registered_types_py.erase(tinfo->type);

            // Actually just `std::erase_if`, but that's only available in C++20
            auto &cache = internals.inactive_override_cache;
            for (auto it = cache.begin(), last = cache.end(); it != last;) {
                if (it->first == (PyObject *) tinfo->type) {
                    it = cache.erase(it);
                } else {
                    ++it;
                }
            }

            delete tinfo;
        }
    });

    type_type_dealloc()(obj);
}

#if defined(PYBIND11_TYPE_CREATION_VIA_SPEC)

PYBIND11_INLINE PyTypeObject *make_default_metaclass() {
    static PyType_Slot slots[]
        = {{Py_tp_call, reinterpret_cast<void *>(pybind11_meta_call)},
           {Py_tp_setattro, reinterpret_cast<void *>(pybind11_meta_setattro)},
           {Py_tp_getattro, reinterpret_cast<void *>(pybind11_meta_getattro)},
           {Py_tp_dealloc, reinterpret_cast<void *>(pybind11_meta_dealloc)},
           {0, nullptr}};
    static PyType_Spec spec = {PYBIND11_DUMMY_MODULE_NAME ".pybind11_type",
                               0, // inherit from type
                               0,
                               Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
                               slots};
    PyObject *type = PyType_FromMetaclass(
        nullptr, nullptr, &spec, reinterpret_cast<PyObject *>(&PyType_Type));
    if (!type) {
        pybind11_fail("make_default_metaclass(): failure in PyType_FromMetaclass(): "
                      + error_string());
    }
    return reinterpret_cast<PyTypeObject *>(type);
}

#else // legacy: fill in a PyHeapTypeObject by hand

PYBIND11_INLINE PyTypeObject *make_default_metaclass() {
    constexpr auto *name = "pybind11_type";
    auto name_obj = reinterpret_steal<object>(PYBIND11_FROM_STRING(name));

    /* Danger zone: from now (and until PyType_Ready), make sure to
       issue no Python C API calls which could potentially invoke the
       garbage collector (the GC will call type_traverse(), which will in
       turn find the newly constructed type in an invalid state) */
    auto *heap_type = reinterpret_cast<PyHeapTypeObject *>(PyType_Type.tp_alloc(&PyType_Type, 0));
    if (!heap_type) {
        pybind11_fail("make_default_metaclass(): error allocating metaclass!");
    }

    heap_type->ht_name = name_obj.inc_ref().ptr();
#    ifdef PYBIND11_BUILTIN_QUALNAME
    heap_type->ht_qualname = name_obj.inc_ref().ptr();
#    endif

    auto *type = &heap_type->ht_type;
    type->tp_name = name;
    type->tp_base = type_incref(&PyType_Type);
    type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;

    type->tp_call = pybind11_meta_call;

    type->tp_setattro = pybind11_meta_setattro;
    type->tp_getattro = pybind11_meta_getattro;

    type->tp_dealloc = pybind11_meta_dealloc;

    if (PyType_Ready(type) < 0) {
        pybind11_fail("make_default_metaclass(): failure in PyType_Ready()!");
    }

    setattr(reinterpret_cast<PyObject *>(type), "__module__", str(PYBIND11_DUMMY_MODULE_NAME));
    PYBIND11_SET_OLDPY_QUALNAME(type, name_obj);

    return type;
}

#endif // PYBIND11_TYPE_CREATION_VIA_SPEC

PYBIND11_INLINE void traverse_offset_bases(void *valueptr,
                                           const detail::type_info *tinfo,
                                           instance *self,
                                           bool (*f)(void * /*parentptr*/, instance * /*self*/)) {
    for (handle h : get_bases(tinfo->type)) {
        if (auto *parent_tinfo = get_type_info(reinterpret_cast<PyTypeObject *>(h.ptr()))) {
            for (auto &c : parent_tinfo->implicit_casts) {
                if (c.first == tinfo->cpptype) {
                    auto *parentptr = c.second(valueptr);
                    if (parentptr != valueptr) {
                        f(parentptr, self);
                    }
                    traverse_offset_bases(parentptr, parent_tinfo, self, f);
                    break;
                }
            }
        }
    }
}

#if defined(Py_GIL_DISABLED) && !defined(PYBIND11_OPAQUE_PYOBJECT)
PYBIND11_INLINE void enable_try_inc_ref(PyObject *obj) {
#    if PY_VERSION_HEX >= 0x030E00A4
    PyUnstable_EnableTryIncRef(obj);
#    else
    if (_Py_IsImmortal(obj)) {
        return;
    }
    for (;;) {
        Py_ssize_t shared = _Py_atomic_load_ssize_relaxed(&obj->ob_ref_shared);
        if ((shared & _Py_REF_SHARED_FLAG_MASK) != 0) {
            // Nothing to do if it's in WEAKREFS, QUEUED, or MERGED states.
            return;
        }
        if (_Py_atomic_compare_exchange_ssize(
                &obj->ob_ref_shared, &shared, shared | _Py_REF_MAYBE_WEAKREF)) {
            return;
        }
    }
#    endif
}

#endif
PYBIND11_INLINE bool register_instance_impl(void *ptr, instance *self) {
    assert(ptr);
#if defined(PYBIND11_OPAQUE_PYOBJECT)
    if (self->registry_weakref == nullptr) {
        self->registry_weakref = PyWeakref_NewRef(instance_object(self), nullptr);
        if (self->registry_weakref == nullptr) {
            throw error_already_set();
        }
    }
#elif defined(Py_GIL_DISABLED)
    enable_try_inc_ref(instance_object(self));
#endif
    with_instance_map(ptr, [&](instance_map &instances) { instances.emplace(ptr, self); });
    return true; // unused, but gives the same signature as the deregister func
}

PYBIND11_INLINE bool deregister_instance_impl(void *ptr, instance *self) {
    assert(ptr);
    return with_instance_map(ptr, [&](instance_map &instances) {
        auto range = instances.equal_range(ptr);
        for (auto it = range.first; it != range.second; ++it) {
            if (self == it->second) {
                instances.erase(it);
                return true;
            }
        }
        return false;
    });
}

PYBIND11_INLINE void register_instance(instance *self, void *valptr, const type_info *tinfo) {
    register_instance_impl(valptr, self);
    if (!tinfo->simple_ancestors) {
        traverse_offset_bases(valptr, tinfo, self, register_instance_impl);
    }
}

PYBIND11_INLINE bool deregister_instance(instance *self, void *valptr, const type_info *tinfo) {
    bool ret = deregister_instance_impl(valptr, self);
    if (!tinfo->simple_ancestors) {
        traverse_offset_bases(valptr, tinfo, self, deregister_instance_impl);
    }
    return ret;
}

#if defined(Py_LIMITED_API)

// Stand-in for PyInstanceMethod_Type, which the stable ABI does not export. Like CPython's
// instancemethod: `__get__` binds through types.MethodType, calls and unknown attributes go to
// the wrapped function. The type is shared through the internals.
struct instancemethod_object {
#    if !defined(PYBIND11_OPAQUE_PYOBJECT)
    PyObject_HEAD
#    endif
    PyObject *func;
};

PYBIND11_INLINE instancemethod_object *instancemethod_data(PyObject *self) {
#    if defined(PYBIND11_OPAQUE_PYOBJECT)
    return static_cast<instancemethod_object *>(PyObject_GetTypeData(self, Py_TYPE(self)));
#    else
    return reinterpret_cast<instancemethod_object *>(self);
#    endif
}

PYBIND11_INLINE PyTypeObject *get_bound_method_type() {
    static PyTypeObject *const type = [] {
        auto types = reinterpret_steal<object>(PyImport_ImportModule("types"));
        if (!types) {
            throw error_already_set();
        }
        // Static builtin type: keep the reference forever.
        return reinterpret_cast<PyTypeObject *>(
            types.attr("MethodType").cast<object>().release().ptr());
    }();
    return type;
}

PYBIND11_INLINE bool is_bound_method(PyObject *obj) {
    return Py_TYPE(obj) == get_bound_method_type();
}

PYBIND11_INLINE PyObject *bound_method_function(PyObject *obj) {
    PyObject *func = PyObject_GetAttrString(obj, "__func__");
    if (func == nullptr) {
        PyErr_Clear();
        return nullptr;
    }
    Py_DECREF(func); // the method object keeps the function alive
    return func;
}

extern "C" PYBIND11_INLINE PyObject *
instancemethod_descr_get(PyObject *self, PyObject *obj, PyObject * /*type*/) {
    if (obj == nullptr || obj == Py_None) {
        Py_INCREF(self);
        return self;
    }
    return PyObject_CallFunctionObjArgs(reinterpret_cast<PyObject *>(get_bound_method_type()),
                                        instancemethod_data(self)->func,
                                        obj,
                                        nullptr);
}

extern "C" PYBIND11_INLINE PyObject *
instancemethod_call(PyObject *self, PyObject *args, PyObject *kwargs) {
    return PyObject_Call(instancemethod_data(self)->func, args, kwargs);
}

extern "C" PYBIND11_INLINE PyObject *instancemethod_getattro(PyObject *self, PyObject *name) {
    // Descriptors of the instancemethod type itself (`__func__`, `__class__`, ...) win;
    // everything else (`__name__`, `__doc__`, `__module__`, ...) comes from the function.
    object descr = type_lookup(Py_TYPE(self), name);
    if (descr) {
        auto get = reinterpret_cast<descrgetfunc>(
            PyType_GetSlot(Py_TYPE(descr.ptr()), Py_tp_descr_get));
        if (get != nullptr) {
            return get(descr.ptr(), self, reinterpret_cast<PyObject *>(Py_TYPE(self)));
        }
    }
    return PyObject_GetAttr(instancemethod_data(self)->func, name);
}

extern "C" PYBIND11_INLINE PyObject *instancemethod_repr(PyObject *self) {
    PyObject *func = instancemethod_data(self)->func;
    auto name = reinterpret_steal<object>(PyObject_GetAttrString(func, "__name__"));
    if (!name) {
        PyErr_Clear();
        return PyUnicode_FromFormat("<instancemethod at %p>", self);
    }
    return PyUnicode_FromFormat("<instancemethod %U at %p>", name.ptr(), self);
}

extern "C" PYBIND11_INLINE int
instancemethod_traverse(PyObject *self, visitproc visit, void *arg) {
    Py_VISIT(instancemethod_data(self)->func);
    Py_VISIT(Py_TYPE(self));
    return 0;
}

extern "C" PYBIND11_INLINE int instancemethod_clear(PyObject *self) {
    Py_CLEAR(instancemethod_data(self)->func);
    return 0;
}

extern "C" PYBIND11_INLINE void instancemethod_dealloc(PyObject *self) {
    PyTypeObject *type = Py_TYPE(self);
    PyObject_GC_UnTrack(self);
    Py_CLEAR(instancemethod_data(self)->func);
    type_free(type, self);
    Py_DECREF(reinterpret_cast<PyObject *>(type));
}

PYBIND11_INLINE PyTypeObject *get_instancemethod_type() {
    return with_internals([](internals &internals) {
        if (internals.instancemethod_type == nullptr) {
            static PyMemberDef members[] = {{"__func__",
                                             Py_T_OBJECT_EX,
                                             offsetof(instancemethod_object, func),
                                             Py_READONLY | PYBIND11_MEMBER_OFFSET_FLAGS,
                                             nullptr},
                                            {nullptr, 0, 0, 0, nullptr}};
            static PyType_Slot slots[]
                = {{Py_tp_descr_get, reinterpret_cast<void *>(instancemethod_descr_get)},
                   {Py_tp_call, reinterpret_cast<void *>(instancemethod_call)},
                   {Py_tp_getattro, reinterpret_cast<void *>(instancemethod_getattro)},
                   {Py_tp_repr, reinterpret_cast<void *>(instancemethod_repr)},
                   {Py_tp_traverse, reinterpret_cast<void *>(instancemethod_traverse)},
                   {Py_tp_clear, reinterpret_cast<void *>(instancemethod_clear)},
                   {Py_tp_dealloc, reinterpret_cast<void *>(instancemethod_dealloc)},
                   {Py_tp_members, reinterpret_cast<void *>(members)},
                   {0, nullptr}};
            static PyType_Spec spec
                = {PYBIND11_DUMMY_MODULE_NAME ".instancemethod",
                   PYBIND11_TYPE_DATA_SIZE(instancemethod_object),
                   0,
                   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_DISALLOW_INSTANTIATION,
                   slots};
            PyObject *type = PyType_FromSpec(&spec);
            if (type == nullptr) {
                pybind11_fail("get_instancemethod_type(): failure in PyType_FromSpec(): "
                              + error_string());
            }
            internals.instancemethod_type = reinterpret_cast<PyTypeObject *>(type);
        }
        return internals.instancemethod_type;
    });
}

PYBIND11_INLINE bool is_instancemethod(PyObject *obj) {
    return Py_TYPE(obj) == get_instancemethod_type();
}

PYBIND11_INLINE PyObject *instancemethod_function(PyObject *obj) {
    return instancemethod_data(obj)->func;
}

PYBIND11_INLINE PyObject *instancemethod_new(PyObject *func) {
    PyTypeObject *type = get_instancemethod_type();
    PyObject *self = type_alloc(type);
    if (self == nullptr) {
        return nullptr;
    }
    Py_INCREF(func);
    instancemethod_data(self)->func = func;
    return self;
}

#endif // Py_LIMITED_API

PYBIND11_INLINE PyObject *type_alloc(PyTypeObject *type) {
#if defined(PYBIND11_HAS_DIRECT_STRUCT_ACCESS)
    return type->tp_alloc(type, 0);
#else
    return reinterpret_cast<allocfunc>(PyType_GetSlot(type, Py_tp_alloc))(type, 0);
#endif
}

PYBIND11_INLINE void type_free(PyTypeObject *type, PyObject *self) {
#if defined(PYBIND11_HAS_DIRECT_STRUCT_ACCESS)
    type->tp_free(self);
#else
    reinterpret_cast<freefunc>(PyType_GetSlot(type, Py_tp_free))(self);
#endif
}

PYBIND11_INLINE PyObject **instance_dict_ptr(PyObject *self) {
#if !defined(Py_LIMITED_API)
    return _PyObject_GetDictPtr(self);
#else
    // Only `py::dynamic_attr()` types have a `__dict__` slot pybind11 manages (recorded in the
    // type_info of the pybind11 type that introduced it); Python subclasses that add a managed
    // dict clear it themselves before calling the base tp_dealloc.
    for (const auto *tinfo : all_type_info(Py_TYPE(self))) {
        if (tinfo->dictoffset > 0) {
            return reinterpret_cast<PyObject **>(reinterpret_cast<char *>(self)
                                                 + tinfo->dictoffset);
        }
    }
    return nullptr;
#endif
}

#if defined(PYBIND11_OPAQUE_PYOBJECT)
PYBIND11_INLINE Py_ssize_t instance_data_offset() {
    // The same for every pybind11 type: the type data of pybind11_object, whose base is object.
    static const Py_ssize_t offset = [] {
        handle base = get_internals().instance_base;
        auto basicsize = base.attr("__basicsize__").cast<Py_ssize_t>();
        return basicsize - PyType_GetTypeDataSize(reinterpret_cast<PyTypeObject *>(base.ptr()));
    }();
    return offset;
}
#endif

PYBIND11_INLINE PyObject *make_new_instance(PyTypeObject *type) {
#if defined(PYPY_VERSION)
    // PyPy gets tp_basicsize wrong (issue 2482) under multiple inheritance when the first
    // inherited object is a plain Python type (i.e. not derived from an extension type).  Fix it.
    ssize_t instance_size = static_cast<ssize_t>(sizeof(instance));
    if (type->tp_basicsize < instance_size) {
        type->tp_basicsize = instance_size;
    }
#endif
    PyObject *self = type_alloc(type);
    auto *inst = get_instance(self);
    // Allocate the value/holder internals:
    inst->allocate_layout();

    return self;
}

extern "C" PYBIND11_INLINE PyObject *
pybind11_object_new(PyTypeObject *type, PyObject *, PyObject *) {
    return make_new_instance(type);
}

extern "C" PYBIND11_INLINE int pybind11_object_init(PyObject *self, PyObject *, PyObject *) {
    PyTypeObject *type = Py_TYPE(self);
    std::string msg = get_fully_qualified_tp_name(type) + ": No constructor defined!";
    set_error(PyExc_TypeError, msg.c_str());
    return -1;
}

PYBIND11_INLINE void add_patient(PyObject *nurse, PyObject *patient) {
    auto *instance = get_instance(nurse);
    instance->has_patients = true;
    Py_INCREF(patient);

    with_internals([&](internals &internals) { internals.patients[nurse].push_back(patient); });
}

PYBIND11_INLINE void clear_patients(PyObject *self) {
    auto *instance = get_instance(self);
    std::vector<PyObject *> patients;

    with_internals([&](internals &internals) {
        auto pos = internals.patients.find(self);

        if (pos == internals.patients.end()) {
            pybind11_fail(
                "FATAL: Internal consistency check failed: Invalid clear_patients() call.");
        }

        // Clearing the patients can cause more Python code to run, which
        // can invalidate the iterator. Extract the vector of patients
        // from the unordered_map first.
        patients = std::move(pos->second);
        internals.patients.erase(pos);
    });

    instance->has_patients = false;
    for (PyObject *&patient : patients) {
        Py_CLEAR(patient);
    }
}

PYBIND11_INLINE void clear_instance(PyObject *self) {
    auto *instance = get_instance(self);

    // Deallocate any values/holders, if present:
    for (auto &v_h : values_and_holders(instance)) {
        if (v_h) {

            // We have to deregister before we call dealloc because, for virtual MI types, we still
            // need to be able to get the parent pointers.
            if (v_h.instance_registered()
                && !deregister_instance(instance, v_h.value_ptr(), v_h.type)) {
                pybind11_fail(
                    "pybind11_object_dealloc(): Tried to deallocate unregistered instance!");
            }

            if (instance->owned || v_h.holder_constructed()) {
                v_h.type->dealloc(v_h);
            }
        } else if (v_h.holder_constructed()) {
            v_h.type->dealloc(v_h); // Disowned instance.
        }
    }
    // Deallocate the value/holder layout internals:
    instance->deallocate_layout();

#if defined(PYBIND11_OPAQUE_PYOBJECT)
    Py_CLEAR(instance->registry_weakref);
#endif
    if (instance->weakrefs) {
        PyObject_ClearWeakRefs(self);
    }

    PyObject **dict_ptr = instance_dict_ptr(self);
    if (dict_ptr) {
        Py_CLEAR(*dict_ptr);
    }

    if (instance->has_patients) {
        clear_patients(self);
    }
}

extern "C" PYBIND11_INLINE void pybind11_object_dealloc(PyObject *self) {
    auto *type = Py_TYPE(self);

    // If this is a GC tracked object, untrack it first
    // Note that the track call is implicitly done by the
    // default tp_alloc, which we never override.
    if (PyType_HasFeature(type, Py_TPFLAGS_HAVE_GC) != 0) {
        PyObject_GC_UnTrack(self);
    }

#if PY_VERSION_HEX >= 0x030D0000 && !defined(Py_LIMITED_API)
    // PyObject_ClearManagedDict() is available from Python 3.13+. It must be
    // called before tp_free() because on Python 3.14+ tp_free no longer
    // implicitly clears the managed dict, which would abandon the refcounts of
    // objects stored in __dict__ of py::dynamic_attr() types, causing permanent
    // memory leaks.
    if (PyType_HasFeature(type, Py_TPFLAGS_MANAGED_DICT)) {
        PyObject_ClearManagedDict(self);
    }
#endif

    clear_instance(self);

    type_free(type, self);

    // This was not needed before Python 3.8 (Python issue 35810)
    // https://github.com/pybind/pybind11/issues/1946
    Py_DECREF(reinterpret_cast<PyObject *>(type));
}

#if defined(PYBIND11_TYPE_CREATION_VIA_SPEC)

PYBIND11_INLINE PyObject *make_object_base_type(PyTypeObject *metaclass) {
    /* Support weak references (needed for the keep_alive feature) */
    static PyMemberDef members[] = {{"__weaklistoffset__",
                                     Py_T_PYSSIZET,
                                     offsetof(instance, weakrefs),
                                     Py_READONLY | PYBIND11_MEMBER_OFFSET_FLAGS,
                                     nullptr},
                                    {nullptr, 0, 0, 0, nullptr}};
    static PyType_Slot slots[]
        = {{Py_tp_new, reinterpret_cast<void *>(pybind11_object_new)},
           {Py_tp_init, reinterpret_cast<void *>(pybind11_object_init)},
           {Py_tp_dealloc, reinterpret_cast<void *>(pybind11_object_dealloc)},
           {Py_tp_members, reinterpret_cast<void *>(members)},
           {0, nullptr}};
    static PyType_Spec spec = {PYBIND11_DUMMY_MODULE_NAME ".pybind11_object",
                               PYBIND11_TYPE_DATA_SIZE(instance),
                               0,
                               Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
                               slots};
    PyObject *type = PyType_FromMetaclass(
        metaclass, nullptr, &spec, reinterpret_cast<PyObject *>(&PyBaseObject_Type));
    if (!type) {
        pybind11_fail("make_object_base_type(): failure in PyType_FromMetaclass(): "
                      + error_string());
    }
    assert(!PyType_HasFeature(reinterpret_cast<PyTypeObject *>(type), Py_TPFLAGS_HAVE_GC));
    return type;
}

#else // legacy: fill in a PyHeapTypeObject by hand

PYBIND11_INLINE PyObject *make_object_base_type(PyTypeObject *metaclass) {
    constexpr auto *name = "pybind11_object";
    auto name_obj = reinterpret_steal<object>(PYBIND11_FROM_STRING(name));

    /* Danger zone: from now (and until PyType_Ready), make sure to
       issue no Python C API calls which could potentially invoke the
       garbage collector (the GC will call type_traverse(), which will in
       turn find the newly constructed type in an invalid state) */
    auto *heap_type = reinterpret_cast<PyHeapTypeObject *>(metaclass->tp_alloc(metaclass, 0));
    if (!heap_type) {
        pybind11_fail("make_object_base_type(): error allocating type!");
    }

    heap_type->ht_name = name_obj.inc_ref().ptr();
#    ifdef PYBIND11_BUILTIN_QUALNAME
    heap_type->ht_qualname = name_obj.inc_ref().ptr();
#    endif

    auto *type = &heap_type->ht_type;
    type->tp_name = name;
    type->tp_base = type_incref(&PyBaseObject_Type);
    type->tp_basicsize = static_cast<ssize_t>(sizeof(instance));
    type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;

    type->tp_new = pybind11_object_new;
    type->tp_init = pybind11_object_init;
    type->tp_dealloc = pybind11_object_dealloc;

    /* Support weak references (needed for the keep_alive feature) */
    type->tp_weaklistoffset = offsetof(instance, weakrefs);

    if (PyType_Ready(type) < 0) {
        pybind11_fail("PyType_Ready failed in make_object_base_type(): " + error_string());
    }

    setattr(reinterpret_cast<PyObject *>(type), "__module__", str(PYBIND11_DUMMY_MODULE_NAME));
    PYBIND11_SET_OLDPY_QUALNAME(type, name_obj);

    assert(!PyType_HasFeature(type, Py_TPFLAGS_HAVE_GC));
    return reinterpret_cast<PyObject *>(heap_type);
}

#endif // PYBIND11_TYPE_CREATION_VIA_SPEC

extern "C" PYBIND11_INLINE int pybind11_traverse(PyObject *self, visitproc visit, void *arg) {
#if PY_VERSION_HEX >= 0x030D0000 && !defined(Py_LIMITED_API)
    int ret = PyObject_VisitManagedDict(self, visit, arg);
    if (ret) {
        return ret;
    }
#else
    if (PyObject **dict = instance_dict_ptr(self)) {
        Py_VISIT(*dict);
    }
#endif
    // https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject.tp_traverse
    Py_VISIT(Py_TYPE(self));
    return 0;
}

extern "C" PYBIND11_INLINE int pybind11_clear(PyObject *self) {
#if PY_VERSION_HEX >= 0x030D0000 && !defined(Py_LIMITED_API)
    PyObject_ClearManagedDict(self);
#else
    if (PyObject **dict = instance_dict_ptr(self)) {
        Py_CLEAR(*dict);
    }
#endif
    return 0;
}

PYBIND11_INLINE PyGetSetDef *dynamic_attr_getset() {
    static PyGetSetDef getset[]
        = {{"__dict__", PyObject_GenericGetDict, PyObject_GenericSetDict, nullptr, nullptr},
           {nullptr, nullptr, nullptr, nullptr, nullptr}};
    return getset;
}

#if !defined(Py_LIMITED_API)
PYBIND11_INLINE void enable_dynamic_attributes(PyHeapTypeObject *heap_type) {
    auto *type = &heap_type->ht_type;
    type->tp_flags |= Py_TPFLAGS_HAVE_GC;
#    ifdef PYBIND11_BACKWARD_COMPATIBILITY_TP_DICTOFFSET
    type->tp_dictoffset = type->tp_basicsize;           // place dict at the end
    type->tp_basicsize += (ssize_t) sizeof(PyObject *); // and allocate enough space for it
#    else
    type->tp_flags |= Py_TPFLAGS_MANAGED_DICT;
#    endif
    type->tp_traverse = pybind11_traverse;
    type->tp_clear = pybind11_clear;
    type->tp_getset = dynamic_attr_getset();
}
#endif // !Py_LIMITED_API

extern "C" PYBIND11_INLINE int pybind11_getbuffer(PyObject *obj, Py_buffer *view, int flags) {
    // Look for a `get_buffer` implementation in this type's info or any bases (following MRO).
    type_info *tinfo = nullptr;
    for (auto type : get_mro(Py_TYPE(obj))) {
        tinfo = get_type_info((PyTypeObject *) type.ptr());
        if (tinfo && tinfo->get_buffer) {
            break;
        }
    }
    if (view == nullptr || !tinfo || !tinfo->get_buffer) {
        if (view) {
            view->obj = nullptr;
        }
        set_error(PyExc_BufferError, "pybind11_getbuffer(): Internal error");
        return -1;
    }
    std::memset(view, 0, sizeof(Py_buffer));
    std::unique_ptr<buffer_info> info = nullptr;
    try {
        info.reset(tinfo->get_buffer(obj, tinfo->get_buffer_data));
    } catch (...) {
        try_translate_exceptions();
        raise_from(PyExc_BufferError, "Error getting buffer");
        return -1;
    }
    if (info == nullptr) {
        pybind11_fail("FATAL UNEXPECTED SITUATION: tinfo->get_buffer() returned nullptr.");
    }

    if ((flags & PyBUF_WRITABLE) == PyBUF_WRITABLE && info->readonly) {
        // view->obj = nullptr;  // Was just memset to 0, so not necessary
        set_error(PyExc_BufferError, "Writable buffer requested for readonly storage");
        return -1;
    }

    // Fill in all the information, and then downgrade as requested by the caller, or raise an
    // error if that's not possible.
    view->itemsize = info->itemsize;
    view->len = view->itemsize;
    for (auto s : info->shape) {
        view->len *= s;
    }
    view->ndim = static_cast<int>(info->ndim);
    view->shape = info->shape.data();
    view->strides = info->strides.data();
    view->readonly = static_cast<int>(info->readonly);
    if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT) {
        view->format = const_cast<char *>(info->format.c_str());
    }

    // Note, all contiguity flags imply PyBUF_STRIDES and lower.
    if ((flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS) {
        if (PyBuffer_IsContiguous(view, 'C') == 0) {
            std::memset(view, 0, sizeof(Py_buffer));
            set_error(PyExc_BufferError,
                      "C-contiguous buffer requested for discontiguous storage");
            return -1;
        }
    } else if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS) {
        if (PyBuffer_IsContiguous(view, 'F') == 0) {
            std::memset(view, 0, sizeof(Py_buffer));
            set_error(PyExc_BufferError,
                      "Fortran-contiguous buffer requested for discontiguous storage");
            return -1;
        }
    } else if ((flags & PyBUF_ANY_CONTIGUOUS) == PyBUF_ANY_CONTIGUOUS) {
        if (PyBuffer_IsContiguous(view, 'A') == 0) {
            std::memset(view, 0, sizeof(Py_buffer));
            set_error(PyExc_BufferError, "Contiguous buffer requested for discontiguous storage");
            return -1;
        }

    } else if ((flags & PyBUF_STRIDES) != PyBUF_STRIDES) {
        // If no strides are requested, the buffer must be C-contiguous.
        // https://docs.python.org/3/c-api/buffer.html#contiguity-requests
        if (PyBuffer_IsContiguous(view, 'C') == 0) {
            std::memset(view, 0, sizeof(Py_buffer));
            set_error(PyExc_BufferError,
                      "C-contiguous buffer requested for discontiguous storage");
            return -1;
        }

        view->strides = nullptr;

        // Since this is a contiguous buffer, it can also pretend to be 1D.
        if ((flags & PyBUF_ND) != PyBUF_ND) {
            view->shape = nullptr;
            view->ndim = 0;
        }
    }

    // Set these after all checks so they don't leak out into the caller, and can be automatically
    // cleaned up on error.
    view->buf = info->ptr;
    view->internal = info.release();
    view->obj = obj;
    Py_INCREF(view->obj);
    return 0;
}

extern "C" PYBIND11_INLINE void pybind11_releasebuffer(PyObject *, Py_buffer *view) {
    delete (buffer_info *) view->internal;
}

#if !defined(Py_LIMITED_API)
PYBIND11_INLINE void enable_buffer_protocol(PyHeapTypeObject *heap_type) {
    heap_type->ht_type.tp_as_buffer = &heap_type->as_buffer;

    heap_type->as_buffer.bf_getbuffer = pybind11_getbuffer;
    heap_type->as_buffer.bf_releasebuffer = pybind11_releasebuffer;
}
#endif // !Py_LIMITED_API

#if defined(PYBIND11_TYPE_CREATION_VIA_SPEC)

PYBIND11_INLINE PyObject *make_new_python_type(const type_record &rec) {
    auto &internals = get_internals();
    auto *metaclass = rec.metaclass.ptr() ? reinterpret_cast<PyTypeObject *>(rec.metaclass.ptr())
                                          : internals.default_metaclass;

    // PyType_FromMetaclass() cannot run a pre-PyType_Ready callback, rejects metaclasses with a
    // custom tp_new, and (correctly) refuses a metaclass that is less derived than the one of
    // the base; the legacy path accepts all three.
#    if !defined(Py_LIMITED_API)
    if (rec.custom_type_setup_callback
        || PyType_GetSlot(metaclass, Py_tp_new) != PyType_GetSlot(&PyType_Type, Py_tp_new)
        || (metaclass != internals.default_metaclass
            && !PyType_IsSubtype(metaclass, internals.default_metaclass))) {
        return make_new_python_type_legacy(rec);
    }
#    else
    // Without the legacy path, a less derived py::metaclass() (e.g. `type`) is resolved to the
    // most derived one by PyType_FromMetaclass(), i.e. the class gets pybind11_type.
    if (PyType_GetSlot(metaclass, Py_tp_new) != PyType_GetSlot(&PyType_Type, Py_tp_new)) {
        pybind11_fail(std::string(rec.name)
                      + ": py::metaclass() must not define __new__ under the stable ABI "
                        "(Py_LIMITED_API)");
    }
#    endif

    object module_ = get_module_name_if_available(rec.scope);
    // Persistent: the type keeps pointing at it as tp_name. PyType_FromMetaclass() derives
    // __module__ from the part before the last dot and warns if there is none.
    const auto *full_name
        = c_str((module_ ? str(module_).cast<std::string>() : PYBIND11_DUMMY_MODULE_NAME) + "."
                + rec.name);

    auto bases = tuple(rec.bases);
    if (bases.empty()) {
        bases = make_tuple(handle(internals.instance_base));
    }
    object base = bases.size() == 1 ? bases[0].cast<object>() : static_cast<object>(bases);

    std::vector<PyType_Slot> slots;
    /* Don't inherit base __init__ */
    slots.push_back({Py_tp_init, reinterpret_cast<void *>(pybind11_object_init)});
    if (rec.doc && options::show_user_defined_docstrings()) {
        slots.push_back({Py_tp_doc, const_cast<char *>(rec.doc)});
    }
    unsigned int flags = Py_TPFLAGS_DEFAULT;
    if (!rec.is_final) {
        flags |= Py_TPFLAGS_BASETYPE;
    }
    int basicsize = 0; // inherit the instance layout from the base
#    if defined(Py_LIMITED_API)
    // No Py_TPFLAGS_MANAGED_DICT in the stable ABI: the `__dict__` slot is appended to the base
    // layout unless a base has one already. (PEP 697 relative offsets are not applied to
    // `__dictoffset__` on 3.12, so the offset is absolute.)
    PyMemberDef dict_member[] = {{"__dictoffset__", Py_T_PYSSIZET, 0, Py_READONLY, nullptr},
                                 {nullptr, 0, 0, 0, nullptr}};
#    endif
    if (rec.dynamic_attr) {
        flags |= Py_TPFLAGS_HAVE_GC;
#    if !defined(Py_LIMITED_API)
        flags |= Py_TPFLAGS_MANAGED_DICT;
#    else
        bool base_has_dict = false;
        int base_basicsize = 0;
        for (handle b : bases) {
            auto *b_type = reinterpret_cast<PyTypeObject *>(b.ptr());
            base_has_dict
                |= type_generic_getattr(b_type, "__dictoffset__").cast<Py_ssize_t>() != 0;
            base_basicsize = std::max(base_basicsize,
                                      type_generic_getattr(b_type, "__basicsize__").cast<int>());
        }
        if (!base_has_dict) {
            dict_member[0].offset = base_basicsize;
            basicsize = base_basicsize + static_cast<int>(sizeof(PyObject *));
            slots.push_back({Py_tp_members, reinterpret_cast<void *>(dict_member)});
        }
#    endif
        slots.push_back({Py_tp_traverse, reinterpret_cast<void *>(pybind11_traverse)});
        slots.push_back({Py_tp_clear, reinterpret_cast<void *>(pybind11_clear)});
        slots.push_back({Py_tp_getset, reinterpret_cast<void *>(dynamic_attr_getset())});
    }
    if (rec.buffer_protocol) {
        slots.push_back({Py_bf_getbuffer, reinterpret_cast<void *>(pybind11_getbuffer)});
        slots.push_back({Py_bf_releasebuffer, reinterpret_cast<void *>(pybind11_releasebuffer)});
    }
    slots.push_back({0, nullptr});

    PyType_Spec spec = {full_name, basicsize, 0, flags, slots.data()};
    PyObject *type = PyType_FromMetaclass(metaclass, nullptr, &spec, base.ptr());
    if (!type) {
        pybind11_fail(std::string(rec.name) + ": PyType_FromMetaclass failed: " + error_string());
    }
    assert(!rec.dynamic_attr
           || PyType_HasFeature(reinterpret_cast<PyTypeObject *>(type), Py_TPFLAGS_HAVE_GC));

    /* Register type with the parent scope */
    if (rec.scope) {
        setattr(rec.scope, rec.name, type);
    } else {
        Py_INCREF(type); // Keep it alive forever (reference leak)
    }

    if (module_) { // Needed by pydoc
        setattr(type, "__module__", module_);
    }
    if (rec.scope && !PyModule_Check(rec.scope.ptr()) && hasattr(rec.scope, "__qualname__")) {
        setattr(type,
                "__qualname__",
                reinterpret_steal<object>(PyUnicode_FromFormat(
                    "%U.%s", rec.scope.attr("__qualname__").ptr(), rec.name)));
    }

    return type;
}

#else

PYBIND11_INLINE PyObject *make_new_python_type(const type_record &rec) {
    return make_new_python_type_legacy(rec);
}

#endif // PYBIND11_TYPE_CREATION_VIA_SPEC

#if !defined(Py_LIMITED_API)
PYBIND11_INLINE PyObject *make_new_python_type_legacy(const type_record &rec) {
    auto name = reinterpret_steal<object>(PYBIND11_FROM_STRING(rec.name));

    auto qualname = name;
    if (rec.scope && !PyModule_Check(rec.scope.ptr()) && hasattr(rec.scope, "__qualname__")) {
        qualname = reinterpret_steal<object>(
            PyUnicode_FromFormat("%U.%U", rec.scope.attr("__qualname__").ptr(), name.ptr()));
    }

    object module_ = get_module_name_if_available(rec.scope);
    const auto *full_name = c_str(
#    if !defined(PYPY_VERSION)
        module_ ? str(module_).cast<std::string>() + "." + rec.name :
#    endif
                rec.name);

    char *tp_doc = nullptr;
    if (rec.doc && options::show_user_defined_docstrings()) {
        /* Allocate memory for docstring (Python will free this later on) */
        size_t size = std::strlen(rec.doc) + 1;
#    if PY_VERSION_HEX >= 0x030D0000
        tp_doc = static_cast<char *>(PyMem_MALLOC(size));
#    else
        tp_doc = (char *) PyObject_MALLOC(size);
#    endif
        std::memcpy((void *) tp_doc, rec.doc, size);
    }

    auto &internals = get_internals();
    auto bases = tuple(rec.bases);
    auto *base = (bases.empty()) ? internals.instance_base : bases[0].ptr();

    /* Danger zone: from now (and until PyType_Ready), make sure to
       issue no Python C API calls which could potentially invoke the
       garbage collector (the GC will call type_traverse(), which will in
       turn find the newly constructed type in an invalid state) */
    auto *metaclass = rec.metaclass.ptr() ? reinterpret_cast<PyTypeObject *>(rec.metaclass.ptr())
                                          : internals.default_metaclass;

    auto *heap_type = reinterpret_cast<PyHeapTypeObject *>(metaclass->tp_alloc(metaclass, 0));
    if (!heap_type) {
        pybind11_fail(std::string(rec.name) + ": Unable to create type object!");
    }

    heap_type->ht_name = name.release().ptr();
#    ifdef PYBIND11_BUILTIN_QUALNAME
    heap_type->ht_qualname = qualname.inc_ref().ptr();
#    endif

    auto *type = &heap_type->ht_type;
    type->tp_name = full_name;
    type->tp_doc = tp_doc;
    type->tp_base = type_incref(reinterpret_cast<PyTypeObject *>(base));
    type->tp_basicsize = static_cast<ssize_t>(sizeof(instance));
    if (!bases.empty()) {
        type->tp_bases = bases.release().ptr();
    }

    /* Don't inherit base __init__ */
    type->tp_init = pybind11_object_init;

    /* Supported protocols */
    type->tp_as_number = &heap_type->as_number;
    type->tp_as_sequence = &heap_type->as_sequence;
    type->tp_as_mapping = &heap_type->as_mapping;
    type->tp_as_async = &heap_type->as_async;

    /* Flags */
    type->tp_flags |= Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE;
    if (!rec.is_final) {
        type->tp_flags |= Py_TPFLAGS_BASETYPE;
    }

    if (rec.dynamic_attr) {
        enable_dynamic_attributes(heap_type);
    }

    if (rec.buffer_protocol) {
        enable_buffer_protocol(heap_type);
    }

    if (rec.custom_type_setup_callback) {
        rec.custom_type_setup_callback(heap_type);
    }

    if (PyType_Ready(type) < 0) {
        pybind11_fail(std::string(rec.name) + ": PyType_Ready failed: " + error_string());
    }

    assert(!rec.dynamic_attr || PyType_HasFeature(type, Py_TPFLAGS_HAVE_GC));

    /* Register type with the parent scope */
    if (rec.scope) {
        setattr(rec.scope, rec.name, reinterpret_cast<PyObject *>(type));
    } else {
        Py_INCREF(type); // Keep it alive forever (reference leak)
    }

    if (module_) { // Needed by pydoc
        setattr(reinterpret_cast<PyObject *>(type), "__module__", module_);
    }

    PYBIND11_SET_OLDPY_QUALNAME(type, qualname);

    return reinterpret_cast<PyObject *>(type);
}
#endif // !Py_LIMITED_API

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
