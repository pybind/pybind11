/*
    tests/test_freezable_type_caster.cpp

    This test verifies that const-ness is propagated through the type_caster framework.
    The returned FreezableInt type is a custom type that is frozen when returned as a
    const T or std::reference<const T>, and not frozen otherwise.
    This test is somewhat complicated because it introduces a custom python type to
    manage the frozen aspect of the type rather then relying on native pybind11 which
    does not support such a feature.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/
#include <pybind11/pybind11.h>

#include <deque>
#include <map>
#include <type_traits>
#include <unordered_map>
#include <cstring>
#include <memory>

#include "pybind11_tests.h"

/// C++ type
class FreezableInt {
 public:
  int64_t x = 1;
};

/// Python wrapper for C++ type which supports mutable/immutable variants.
typedef struct PyFreezableInt {
  PyObject_HEAD

  FreezableInt* value;

  bool is_immutable;
  bool is_owned;
} PyFreezableInt;

PyObject* PyFreezableInt_getattro(PyObject* self, PyObject* name) {
  auto* freezable = reinterpret_cast<PyFreezableInt*>(self);
  assert(freezable != nullptr);

  const char* attr = nullptr;
  if (PyBytes_Check(name)) {
    attr = PyBytes_AsString(name);
  }
#if PY_VERSION_HEX > 0x03030000
  if (PyUnicode_Check(name)) {
    attr = PyUnicode_AsUTF8(name);
  }
#endif
  if (attr != nullptr && strcmp(attr, "x") == 0) {
    return PyLong_FromLongLong(static_cast<long long>(freezable->value->x));
  }
  return PyObject_GenericGetAttr(self, name);
}

int PyFreezableInt_setattro(PyObject* self, PyObject* name, PyObject* value) {
  auto* freezable = reinterpret_cast<PyFreezableInt*>(self);
  assert(freezable != nullptr);

  const char* attr = nullptr;
  if (PyBytes_Check(name)) {
    attr = PyBytes_AsString(name);
  }
#if PY_VERSION_HEX > 0x03030000
  if (PyUnicode_Check(name)) {
    attr = PyUnicode_AsUTF8(name);
  }
#endif
  if (attr != nullptr && strcmp(attr, "x") == 0) {
    if (freezable->is_immutable) {
      PyErr_Format(PyExc_ValueError, "Instance is immutable; cannot set values");
      return -1;
    }

    if (PyLong_Check(value)) {
      freezable->value->x = static_cast<int64_t>(PyLong_AsLongLong(value));
      return 0;
    }
#if PY_VERSION_HEX < 0x03000000
    if (PyInt_Check(value)) {
      freezable->value->x = static_cast<int64_t>(PyInt_AsLong(value));
      return 0;
    }
#endif
    // "Cannot set a non-numeric value of type %s"
    PyErr_SetObject(PyExc_ValueError, value);
    return -1;
  }

  return PyObject_GenericSetAttr(self, name, value);
}

void PyFreezableInt_dealloc(PyObject* self);


/* https://github.com/cython/cython/issues/3474 */
#if defined(__GNUC__) || defined(__clang__) && PY_VERSION_HEX >= 0x030800b4 && PY_VERSION_HEX < 0x03090000
#define PY38_WARNING_WORKAROUND_ENABLED 1
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

static PyTypeObject PyFreezableInt_Type{
        PyVarObject_HEAD_INIT(nullptr, 0) /**/
        "pybind11_tests.test_freezable.FreezableInt",  /* tp_name */
        sizeof(PyFreezableInt),                      /* tp_basicsize */
    0,                                          /* tp_itemsize */
    &PyFreezableInt_dealloc,                         /* tp_dealloc */
    0,                                          /* tp_print | tp_vectorcall_offset */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_as_async */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    &PyFreezableInt_getattro,                   /* tp_getattro */
    &PyFreezableInt_setattro,                   /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                         /* tp_flags */
    "FreezableInt objects",                     /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    0,                                          /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    0,                                          /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
    0,                                          /* tp_version_tag */
#if PY_VERSION_HEX >= 0x030400a1
    0,                                          /* tp_finalize */
#endif
#if PY_VERSION_HEX >= 0x030800b1
    0,                                          /* tp_vectorcall */
#endif
#if PY_VERSION_HEX >= 0x030800b4 && PY_VERSION_HEX < 0x03090000
    0,                                          /* tp_print */
#endif
#if defined(PYPY_VERSION)
    0,                                          /* tp_pypy_flags */
#endif
};


#if defined(PY38_WARNING_WORKAROUND_ENABLED)
#undef PY38_WARNING_WORKAROUND_ENABLED
#pragma GCC diagnostic pop
#endif

static std::unordered_map<FreezableInt*, void*> mapping;

void PyFreezableInt_dealloc(PyObject* self) {
  auto* freezable = reinterpret_cast<PyFreezableInt*>(self);
  auto it = mapping.find(freezable->value);
  if (it != mapping.end()) {
    mapping.erase(it);
  }
  if (freezable->is_owned) {
    delete freezable->value;
  }
  Py_TYPE(self)->tp_free(self);
}

PyObject* PyFreezableInt_new(FreezableInt* value, bool is_immutable) {
  if (PyType_Ready(&PyFreezableInt_Type) != 0) {
    return nullptr;
  }
  PyFreezableInt* self = PyObject_New(PyFreezableInt, &PyFreezableInt_Type);
  if (!self) {
    return nullptr;
  }
  mapping.emplace(value, self);
  self->value = value;
  self->is_owned = false;
  self->is_immutable = is_immutable;
  return reinterpret_cast<PyObject*>(self);
}

PyObject* PyFreezableInt_new(std::unique_ptr<FreezableInt> value) {
  PyObject* self = PyFreezableInt_new(value.get(), false);
  if (self != nullptr) {
    auto* freezable = reinterpret_cast<PyFreezableInt*>(self);
    freezable->value = value.release();
    freezable->is_owned = true;
  }
  return self;
}


PyObject* PyFreezableInt_reference(FreezableInt* value, bool is_immutable) {
  auto it = mapping.find(value);
  if (it != mapping.end()) {
    auto* freezable = static_cast<PyFreezableInt*>(it->second);
    if (!is_immutable) {
      // We have a single cache of C++ pointer to python object, so if any of the
      // objects becomes immutable, they all become immutable.
      freezable->is_immutable = false;
    }
    Py_INCREF(freezable);
    return reinterpret_cast<PyObject*>(freezable);
  }
  return PyFreezableInt_new(value, is_immutable);
}

/// pybind11 type_caster which returns custom PyFreezableInt python wrapper instead
/// of a pybind11 generated type; this is used to make const* and const& immutable
/// in python.
namespace pybind11 {
namespace detail {

template <>
struct type_caster<FreezableInt> {
  static constexpr auto name = _<FreezableInt>();

  // C++ -> Python
  // ... construct an immutable python type referencing src.  This isn't really
  // available in pybind11 and would have to be built by hand.
  static handle cast(const FreezableInt* src, return_value_policy, handle) {
    if (!src) return pybind11::none().release();
    return PyFreezableInt_reference(const_cast<FreezableInt*>(src), true);
  }
  static handle cast(const FreezableInt& src, return_value_policy policy,
                     handle parent) {
    return cast(&src, policy, parent);
  }

  // ... construct a mutable python type referencing src. This is the default
  // pybind11 path.
  static handle cast(FreezableInt* src, return_value_policy, handle) {
    if (!src) return pybind11::none().release();
    return PyFreezableInt_reference(src, false);
  }
  static handle cast(FreezableInt& src, return_value_policy policy, handle parent) {
    return cast(&src, policy, parent);
  }

  // ... construct a mutable python type owning src.
  static handle cast(FreezableInt&& src, return_value_policy, handle) {
    std::unique_ptr<FreezableInt> ptr(new FreezableInt(std::move(src)));
    return PyFreezableInt_new(std::move(ptr));
  }

  type_caster() = default;
  type_caster(type_caster&&) = default;
  type_caster(const type_caster& other) : freezable(other.freezable) {
    if (freezable) Py_INCREF(freezable);
  }

  type_caster& operator=(type_caster&&) = default;
  type_caster& operator=(const type_caster& other) {
    if (freezable) Py_DECREF(freezable);
    freezable = other.freezable;
    if (freezable) Py_INCREF(freezable);
    return *this;
  }

  ~type_caster() {
    if (freezable) Py_DECREF(freezable);
  }

  // Convert Python->C++.
  // ... Merely capture the PyFreezableInt pointer and do additional work in the
  // conversion operator.
  bool load(handle src, bool) {
    if (!PyObject_TypeCheck(src.ptr(), &PyFreezableInt_Type)) {
      return false;
    }
    assert(freezable == nullptr);
    freezable = reinterpret_cast<PyFreezableInt*>(src.ptr());
    Py_INCREF(freezable);
    return true;
  }

  // cast_op_type determines which operator overload to call for a given c++
  // input parameter type. In this case we want to propagate const, etc.
  template <typename T_>
  using cast_op_type = conditional_t<
      std::is_same<remove_reference_t<T_>, const FreezableInt*>::value, const FreezableInt*,
      conditional_t<
          std::is_same<remove_reference_t<T_>, FreezableInt*>::value, FreezableInt*,
          conditional_t<std::is_same<T_, const FreezableInt&>::value, const FreezableInt&,
                        conditional_t<std::is_same<T_, FreezableInt&>::value, FreezableInt&,
                                      /*default is T&&*/ T_>>>>;

  // PYBIND11_TYPE_CASTER
  operator const FreezableInt*() { return freezable->value; }
  operator const FreezableInt&() {
    if (!freezable || !freezable->value) throw reference_cast_error();
    return *freezable->value;
  }
  operator FreezableInt*() {
    if (freezable->is_immutable) throw reference_cast_error();
    return freezable->value;
  }
  operator FreezableInt&() {
    if (!freezable || !freezable->value) throw reference_cast_error();
    if (freezable->is_immutable) throw reference_cast_error();
    return *freezable->value;
  }
  operator FreezableInt&&() && {
    if (!freezable || !freezable->value) throw reference_cast_error();
    owned = *freezable->value;
    return std::move(owned);
  }

 protected:
  PyFreezableInt* freezable = nullptr;
  FreezableInt owned;
};

}  // namespace detail
}  // namespace pybind11

/// C++ module using pybind11 type_caster<FreezableInt> returning mutable/immutable
/// objects.

static FreezableInt shared;  // returned by non-const get.
static FreezableInt shared_const;  // returned by const get.

TEST_SUBMODULE(test_freezable_type_caster, m) {

  Py_INCREF(&PyFreezableInt_Type);

  m.def("make", []() -> FreezableInt { return FreezableInt{}; });
  m.def("reset", [](int x) { shared.x = x; shared_const.x = x; });

  // it takes and it takes and it takes
  m.def("take", [](FreezableInt c) {
    c.x++;
    return c.x;
  });

  m.def("take_ptr", [](FreezableInt* c) {
    c->x++;
    return c->x;
  });
  m.def("take_ref", [](FreezableInt& c) {
    c.x++;
    return c.x;
  });
  m.def("take_wrap", [](std::reference_wrapper<FreezableInt> c) {
    c.get().x++;
    return c.get().x;
  });

  m.def("take_const_ptr", [](const FreezableInt* c) { return 10 * c->x; });
  m.def("take_const_ref", [](const FreezableInt& c) { return 10 * c.x; });
  m.def("take_const_wrap",
        [](std::reference_wrapper<const FreezableInt> c) { return 10 * c.get().x; });

  m.def("get", []() -> FreezableInt { return shared; });

  m.def("get_ptr", []() -> FreezableInt* { return &shared; });
  m.def("get_ref", []() -> FreezableInt& { return shared; });
  m.def("get_wrap",
        []() -> std::reference_wrapper<FreezableInt> { return std::ref(shared); });

  m.def("get_const_ptr", []() -> const FreezableInt* {
    shared_const.x++;
    return &shared_const;
  });
  m.def("get_const_ref", []() -> const FreezableInt& {
    shared_const.x++;
    return shared_const;
  });
  m.def("get_const_wrap", []() -> std::reference_wrapper<const FreezableInt> {
    shared_const.x++;
    return std::cref(shared_const);
  });

  m.def("roundtrip", [](FreezableInt c) -> FreezableInt {
    c.x++;
    return c;
  });
  m.def("roundtrip_ptr", [](FreezableInt* c) -> FreezableInt* {
    c->x++;
    return c;
  });
  m.def("roundtrip_ref", [](FreezableInt& c) -> FreezableInt& {
    c.x++;
    return c;
  });
  m.def("roundtrip_wrap",
        [](std::reference_wrapper<FreezableInt> c) -> std::reference_wrapper<FreezableInt> {
          c.get().x++;
          return c;
        });

  m.def("roundtrip_const_ref", [](const FreezableInt& c) -> std::reference_wrapper<const FreezableInt> {
    return std::cref(c);
  });

}
