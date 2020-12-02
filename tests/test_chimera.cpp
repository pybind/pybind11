/*
    tests/test_chimera.cpp -- This demonstrates a hybrid usage of pybind11, where the
    type caster returns a hand-rolled python object type rather than relying on the
    natural python bindings.

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
class Chimera {
 public:
  int64_t x = 1;
};

/// Python wrapper for C++ type which supports mutable/immutable variants.
typedef struct PyChimera {
  PyObject_HEAD

  Chimera* value;

  bool is_immutable;
  bool is_owned;
} PyChimera;

PyObject* PyChimera_getattro(PyObject* self, PyObject* name) {
  auto* chimera = reinterpret_cast<PyChimera*>(self);
  assert(chimera != nullptr);

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
    return PyLong_FromLongLong(static_cast<long long>(chimera->value->x));
  }
  return PyObject_GenericGetAttr(self, name);
}

int PyChimera_setattro(PyObject* self, PyObject* name, PyObject* value) {
  auto* chimera = reinterpret_cast<PyChimera*>(self);
  assert(chimera != nullptr);

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
    if (chimera->is_immutable) {
      PyErr_Format(PyExc_ValueError, "Instance is immutable; cannot set values");
      return -1;
    }

    if (PyLong_Check(value)) {
      chimera->value->x = static_cast<int64_t>(PyLong_AsLongLong(value));
      return 0;
    }
#if PY_VERSION_HEX < 0x03000000
    if (PyInt_Check(value)) {
      chimera->value->x = static_cast<int64_t>(PyInt_AsLong(value));
      return 0;
    }
#endif
    // "Cannot set a non-numeric value of type %s"
    PyErr_SetObject(PyExc_ValueError, value);
    return -1;
  }

  return PyObject_GenericSetAttr(self, name, value);
}

void PyChimera_dealloc(PyObject* self);


/* https://github.com/cython/cython/issues/3474 */
#if defined(__GNUC__) || defined(__clang__) && PY_VERSION_HEX >= 0x030800b4 && PY_VERSION_HEX < 0x03090000
#define PY38_WARNING_WORKAROUND_ENABLED 1
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

static PyTypeObject PyChimera_Type{
        PyVarObject_HEAD_INIT(nullptr, 0) /**/
        "pybind11_tests.test_chimera.Chimera",  /* tp_name */
        sizeof(PyChimera),                      /* tp_basicsize */
    0,                                          /* tp_itemsize */
    &PyChimera_dealloc,                         /* tp_dealloc */
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
    &PyChimera_getattro,                        /* tp_getattro */
    &PyChimera_setattro,                        /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                         /* tp_flags */
    "Chimera objects",                          /* tp_doc */
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

static std::unordered_map<Chimera*, void*>* mapping =
    new std::unordered_map<Chimera*, void*>();

void PyChimera_dealloc(PyObject* self) {
  auto* chimera = reinterpret_cast<PyChimera*>(self);
  auto it = mapping->find(chimera->value);
  if (it != mapping->end()) {
    mapping->erase(it);
  }
  if (chimera->is_owned) {
    delete chimera->value;
  }
  Py_TYPE(self)->tp_free(self);
}

PyObject* PyChimera_new(Chimera* value, bool is_immutable) {
  if (PyType_Ready(&PyChimera_Type) != 0) {
    return nullptr;
  }
  PyChimera* self = PyObject_New(PyChimera, &PyChimera_Type);
  if (!self) {
    return nullptr;
  }
  mapping->emplace(value, self);
  self->value = value;
  self->is_owned = false;
  self->is_immutable = is_immutable;
  return reinterpret_cast<PyObject*>(self);
}

PyObject* PyChimera_new(std::unique_ptr<Chimera> value) {
  PyObject* self = PyChimera_new(value.get(), false);
  if (self != nullptr) {
    auto* chimera = reinterpret_cast<PyChimera*>(self);
    chimera->value = value.release();
    chimera->is_owned = true;
  }
  return self;
}


PyObject* PyChimera_reference(Chimera* value, bool is_immutable) {
  auto it = mapping->find(value);
  if (it != mapping->end()) {
    auto* chimera = static_cast<PyChimera*>(it->second);
    if (!is_immutable) {
      // We have a single cache of C++ pointer to python object, so if any of the
      // objects becomes immutable, they all become immutable.
      chimera->is_immutable = false;
    }
    Py_INCREF(chimera);
    return reinterpret_cast<PyObject*>(chimera);
  }
  return PyChimera_new(value, is_immutable);
}

/// pybind11 type_caster which returns custom PyChimera python wrapper instead
/// of a pybind11 generated type; this is used to make const* and const& immutable
/// in python.
namespace pybind11 {
namespace detail {

template <>
struct type_caster<Chimera> {
  static constexpr auto name = _<Chimera>();

  // C++ -> Python
  // ... construct an immutable python type referencing src.  This isn't really
  // available in pybind11 and would have to be built by hand.
  static handle cast(const Chimera* src, return_value_policy, handle) {
    if (!src) return pybind11::none().release();
    return PyChimera_reference(const_cast<Chimera*>(src), true);
  }
  static handle cast(const Chimera& src, return_value_policy policy,
                     handle parent) {
    return cast(&src, policy, parent);
  }

  // ... construct a mutable python type referencing src. This is the default
  // pybind11 path.
  static handle cast(Chimera* src, return_value_policy, handle) {
    if (!src) return pybind11::none().release();
    return PyChimera_reference(src, false);
  }
  static handle cast(Chimera& src, return_value_policy policy, handle parent) {
    return cast(&src, policy, parent);
  }

  // ... construct a mutable python type owning src.
  static handle cast(Chimera&& src, return_value_policy, handle) {
    std::unique_ptr<Chimera> ptr(new Chimera(std::move(src)));
    return PyChimera_new(std::move(ptr));
  }

  ~type_caster() {
    if (chimera) PyDECREF(chimera);
  }
  // Convert Python->C++.
  // ... Merely capture the PyChimera pointer and do additional work in the
  // conversion operator.
  bool load(handle src, bool) {
    if (!PyObject_TypeCheck(src.ptr(), &PyChimera_Type)) {
      return false;
    }
    assert(chimera == nullptr);
    chimera = reinterpret_cast<PyChimera*>(src.ptr());
    Py_INCREF(chimera);
    return true;
  }

  // cast_op_type determines which operator overload to call for a given c++
  // input parameter type. In this case we want to propagate const, etc.
  template <typename T_>
  using cast_op_type = conditional_t<
      std::is_same<remove_reference_t<T_>, const Chimera*>::value, const Chimera*,
      conditional_t<
          std::is_same<remove_reference_t<T_>, Chimera*>::value, Chimera*,
          conditional_t<std::is_same<T_, const Chimera&>::value, const Chimera&,
                        conditional_t<std::is_same<T_, Chimera&>::value, Chimera&,
                                      /*default is T&&*/ T_>>>>;

  // PYBIND11_TYPE_CASTER
  operator const Chimera*() { return chimera->value; }
  operator const Chimera&() {
    if (!chimera || !chimera->value) throw reference_cast_error();
    return *chimera->value;
  }
  operator Chimera*() {
    if (chimera->is_immutable) throw reference_cast_error();
    return chimera->value;
  }
  operator Chimera&() {
    if (!chimera || !chimera->value) throw reference_cast_error();
    if (chimera->is_immutable) throw reference_cast_error();
    return *chimera->value;
  }
  operator Chimera&&() && {
    if (!chimera || !chimera->value) throw reference_cast_error();
    owned = *chimera->value;
    return std::move(owned);
  }

 protected:
  const PyChimera* chimera = nullptr;
  Chimera owned;
};

}  // namespace detail
}  // namespace pybind11

/// C++ module using pybind11 type_caster<Chimera> returning mutable/immutable
/// objects.

static Chimera* shared = new Chimera();
static Chimera* shared_const = new Chimera();

TEST_SUBMODULE(test_chimera, m) {

  Py_INCREF(&PyChimera_Type);

  m.def("make", []() -> Chimera { return Chimera{}; });
  m.def("reset", [](int x) { shared->x = x; shared_const->x = x; });

  // it takes and it takes and it takes
  m.def("take", [](Chimera c) {
    c.x++;
    return c.x;
  });

  m.def("take_ptr", [](Chimera* c) {
    c->x++;
    return c->x;
  });
  m.def("take_ref", [](Chimera& c) {
    c.x++;
    return c.x;
  });
  m.def("take_wrap", [](std::reference_wrapper<Chimera> c) {
    c.get().x++;
    return c.get().x;
  });

  m.def("take_const_ptr", [](const Chimera* c) { return 10 * c->x; });
  m.def("take_const_ref", [](const Chimera& c) { return 10 * c.x; });
  m.def("take_const_wrap",
        [](std::reference_wrapper<const Chimera> c) { return 10 * c.get().x; });

  m.def("get", []() -> Chimera { return *shared; });

  m.def("get_ptr", []() -> Chimera* { return shared; });
  m.def("get_ref", []() -> Chimera& { return *shared; });
  m.def("get_wrap",
        []() -> std::reference_wrapper<Chimera> { return std::ref(*shared); });
  m.def("get_const_ptr", []() -> const Chimera* {
    shared_const->x++;
    return shared_const;
  });
  m.def("get_const_ref", []() -> const Chimera& {
    shared_const->x++;
    return *shared_const;
  });
  m.def("get_const_wrap", []() -> std::reference_wrapper<const Chimera> {
    shared_const->x++;
    return std::cref(*shared_const);
  });

  m.def("roundtrip", [](Chimera c) -> Chimera {
    c.x++;
    return c;
  });
  m.def("roundtrip_ptr", [](Chimera* c) -> Chimera* {
    c->x++;
    return c;
  });
  m.def("roundtrip_ref", [](Chimera& c) -> Chimera& {
    c.x++;
    return c;
  });
  m.def("roundtrip_wrap",
        [](std::reference_wrapper<Chimera> c) -> std::reference_wrapper<Chimera> {
          c.get().x++;
          return c;
        });
}
