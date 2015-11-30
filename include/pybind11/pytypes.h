/*
    pybind11/typeid.h: Convenience wrapper classes for basic Python types

    Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"
#include <utility>
#include <type_traits>

NAMESPACE_BEGIN(pybind11)

/* A few forward declarations */
class object;
class str;
class object;
class dict;
class iterator;
namespace detail { class accessor; }

/// Holds a reference to a Python object (no reference counting)
class handle {
public:
    handle() : m_ptr(nullptr) { }
    handle(const handle &other) : m_ptr(other.m_ptr) { }
    handle(PyObject *ptr) : m_ptr(ptr) { }
    PyObject *ptr() { return m_ptr; }
    const PyObject *ptr() const { return m_ptr; }
    void inc_ref() const { Py_XINCREF(m_ptr); }
    void dec_ref() const { Py_XDECREF(m_ptr); }
    int ref_count() const { return (int) Py_REFCNT(m_ptr); }
    handle get_type() { return (PyObject *) Py_TYPE(m_ptr); }
    inline iterator begin();
    inline iterator end();
    inline detail::accessor operator[](handle key);
    inline detail::accessor operator[](const char *key);
    inline detail::accessor attr(handle key);
    inline detail::accessor attr(const char *key);
    inline pybind11::str str() const;
    template <typename T> T cast();
    template <typename ... Args> object call(Args&&... args_);
    operator bool() const { return m_ptr != nullptr; }
protected:
    PyObject *m_ptr;
};

/// Holds a reference to a Python object (with reference counting)
class object : public handle {
public:
    object() { }
    object(const object &o) : handle(o) { inc_ref(); }
    object(const handle &h, bool borrowed) : handle(h) { if (borrowed) inc_ref(); }
    object(PyObject *ptr, bool borrowed) : handle(ptr) { if (borrowed) inc_ref(); }
    object(object &&other) { m_ptr = other.m_ptr; other.m_ptr = nullptr; }
    ~object() { dec_ref(); }

    object& operator=(object &other) {
        Py_XINCREF(other.m_ptr);
        Py_XDECREF(m_ptr);
        m_ptr = other.m_ptr;
        return *this;
    }

    object& operator=(object &&other) {
        if (this != &other) {
            PyObject *temp = m_ptr;
            m_ptr = other.m_ptr;
            other.m_ptr = nullptr;
            Py_XDECREF(temp);
        }
        return *this;
    }
};

class iterator : public object {
public:
    iterator(PyObject *obj, bool borrowed = false) : object(obj, borrowed) { ++*this; }
    iterator& operator++() {
        if (ptr())
            value = object(PyIter_Next(ptr()), false);
        return *this;
    }
    bool operator==(const iterator &it) const { return *it == **this; }
    bool operator!=(const iterator &it) const { return *it != **this; }
    object operator*() { return value; }
    const object &operator*() const { return value; }
    bool check() const { return PyIter_Check(ptr()); }
private:
    object value;
};

NAMESPACE_BEGIN(detail)
inline PyObject *get_function(PyObject *value) {
    if (value == nullptr)
        return nullptr;
#if PY_MAJOR_VERSION >= 3
    if (PyInstanceMethod_Check(value))
        value = PyInstanceMethod_GET_FUNCTION(value);
#endif
    if (PyMethod_Check(value))
        value = PyMethod_GET_FUNCTION(value);
    return value;
}

class accessor {
public:
    accessor(PyObject *obj, PyObject *key, bool attr)
        : obj(obj), key(key), attr(attr) { Py_INCREF(key);  }
    accessor(PyObject *obj, const char *key, bool attr)
        : obj(obj), key(PyUnicode_FromString(key)), attr(attr) { }
    accessor(const accessor &a) : obj(a.obj), key(a.key), attr(a.attr)
        { Py_INCREF(key); }
    ~accessor() { Py_DECREF(key); }

    void operator=(accessor o) { operator=(object(o)); }

    void operator=(const handle &h) {
        if (attr) {
            if (PyObject_SetAttr(obj, key, (PyObject *) h.ptr()) < 0)
                throw std::runtime_error("Unable to set object attribute");
        } else {
            if (PyObject_SetItem(obj, key, (PyObject *) h.ptr()) < 0)
                throw std::runtime_error("Unable to set object item");
        }
    }

    operator object() const {
        object result(attr ? PyObject_GetAttr(obj, key)
                           : PyObject_GetItem(obj, key), false);
        if (!result) PyErr_Clear();
        return result;
    }

    operator bool() const {
        if (attr) {
            return (bool) PyObject_HasAttr(obj, key);
        } else {
            object result(PyObject_GetItem(obj, key), false);
            if (!result) PyErr_Clear();
            return (bool) result;
        }
    };

private:
    PyObject *obj;
    PyObject *key;
    bool attr;
};

struct list_accessor {
public:
    list_accessor(PyObject *list, size_t index) : list(list), index(index) { }
    void operator=(list_accessor o) { return operator=(object(o)); }
    void operator=(const handle &o) {
        o.inc_ref(); // PyList_SetItem steals a reference
        if (PyList_SetItem(list, (ssize_t) index, (PyObject *) o.ptr()) < 0)
            throw std::runtime_error("Unable to assign value in Python list!");
    }
    operator object() const {
        PyObject *result = PyList_GetItem(list, (ssize_t) index);
        if (!result)
            throw std::runtime_error("Unable to retrieve value from Python list!");
        return object(result, true);
    }
private:
    PyObject *list;
    size_t index;
};

struct tuple_accessor {
public:
    tuple_accessor(PyObject *tuple, size_t index) : tuple(tuple), index(index) { }
    void operator=(tuple_accessor o) { return operator=(object(o)); }
    void operator=(const handle &o) {
        o.inc_ref(); // PyTuple_SetItem steals a reference
        if (PyTuple_SetItem(tuple, (ssize_t) index, (PyObject *) o.ptr()) < 0)
            throw std::runtime_error("Unable to assign value in Python tuple!");
    }
    operator object() const {
        PyObject *result = PyTuple_GetItem(tuple, (ssize_t) index);
        if (!result)
            throw std::runtime_error("Unable to retrieve value from Python tuple!");
        return object(result, true);
    }
private:
    PyObject *tuple;
    size_t index;
};

struct dict_iterator {
public:
    dict_iterator(PyObject *dict = nullptr, ssize_t pos = -1) : dict(dict), pos(pos) { }
    dict_iterator& operator++() {
        if (!PyDict_Next(dict, &pos, &key, &value))
            pos = -1;
        return *this;
    }
    std::pair<object, object> operator*() {
        return std::make_pair(object(key, true), object(value, true));
    }
    bool operator==(const dict_iterator &it) const { return it.pos == pos; }
    bool operator!=(const dict_iterator &it) const { return it.pos != pos; }
private:
    PyObject *dict, *key, *value;
    ssize_t pos = 0;
};

#if PY_MAJOR_VERSION >= 3
using ::PyLong_AsUnsignedLongLong;
using ::PyLong_AsLongLong;
#else
inline long long PyLong_AsLongLong(PyObject *o) {
    if (PyInt_Check(o)) /// workaround: PyLong_AsLongLong doesn't accept 'int' on Python 2.x
        return (long long) PyLong_AsLong(o);
    else
        return ::PyLong_AsLongLong(o);
}

inline unsigned long long PyLong_AsUnsignedLongLong(PyObject *o) {
    if (PyInt_Check(o)) /// workaround: PyLong_AsUnsignedLongLong doesn't accept 'int' on Python 2.x
        return (unsigned long long) PyLong_AsUnsignedLong(o);
    else
        return ::PyLong_AsUnsignedLongLong(o);
}
#endif

NAMESPACE_END(detail)

inline detail::accessor handle::operator[](handle key) { return detail::accessor(ptr(), key.ptr(), false); }
inline detail::accessor handle::operator[](const char *key) { return detail::accessor(ptr(), key, false); }
inline detail::accessor handle::attr(handle key) { return detail::accessor(ptr(), key.ptr(), true); }
inline detail::accessor handle::attr(const char *key) { return detail::accessor(ptr(), key, true); }
inline iterator handle::begin() { return iterator(PyObject_GetIter(ptr())); }
inline iterator handle::end() { return iterator(nullptr); }

#define PYBIND11_OBJECT_CVT(Name, Parent, CheckFun, CvtStmt) \
    Name(const handle &h, bool borrowed) : Parent(h, borrowed) { CvtStmt; } \
    Name(const object& o): Parent(o) { CvtStmt; } \
    Name(object&& o): Parent(std::move(o)) { CvtStmt; } \
    Name& operator=(object&& o) { return static_cast<Name&>(object::operator=(std::move(o))); CvtStmt; } \
    Name& operator=(object& o) { return static_cast<Name&>(object::operator=(o)); CvtStmt; } \
    bool check() const { return m_ptr != nullptr && (bool) CheckFun(m_ptr); }

#define PYBIND11_OBJECT(Name, Parent, CheckFun) \
    PYBIND11_OBJECT_CVT(Name, Parent, CheckFun, )

#define PYBIND11_OBJECT_DEFAULT(Name, Parent, CheckFun) \
    PYBIND11_OBJECT(Name, Parent, CheckFun) \
    Name() : Parent() { }

class str : public object {
public:
    PYBIND11_OBJECT_DEFAULT(str, object, PyUnicode_Check)
    str(const char *s) : object(PyUnicode_FromString(s), false) { }
    operator const char *() const {
#if PY_MAJOR_VERSION >= 3
        return PyUnicode_AsUTF8(m_ptr);
#else
        m_temp = object(PyUnicode_AsUTF8String(m_ptr), false);
        if (m_temp.ptr() == nullptr)
            return nullptr;
        return PyString_AsString(m_temp.ptr());
#endif
    }
private:
#if PY_MAJOR_VERSION < 3
    mutable object m_temp;
#endif
};

inline pybind11::str handle::str() const {
    PyObject *str = PyObject_Str(m_ptr);
#if PY_MAJOR_VERSION < 3
    PyObject *unicode = PyUnicode_FromEncodedObject(str, "utf-8", nullptr);
    Py_XDECREF(str); str = unicode;
#endif
    return pybind11::str(str, false);
}

class bool_ : public object {
public:
    PYBIND11_OBJECT_DEFAULT(bool_, object, PyBool_Check)
    bool_(bool value) : object(value ? Py_True : Py_False, true) { }
    operator bool() const { return m_ptr && PyLong_AsLong(m_ptr) != 0; }
};

class int_ : public object {
public:
    PYBIND11_OBJECT_DEFAULT(int_, object, PyLong_Check)
    template <typename T,
              typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
    int_(T value) {
        if (sizeof(T) <= sizeof(long)) {
            if (std::is_signed<T>::value)
                m_ptr = PyLong_FromLong((long) value);
            else
                m_ptr = PyLong_FromUnsignedLong((unsigned long) value);
        } else {
            if (std::is_signed<T>::value)
                m_ptr = PyLong_FromLongLong((long long) value);
            else
                m_ptr = PyLong_FromUnsignedLongLong((unsigned long long) value);
        }
    }

    template <typename T,
              typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
    operator T() const {
        if (sizeof(T) <= sizeof(long)) {
            if (std::is_signed<T>::value)
                return (T) PyLong_AsLong(m_ptr);
            else
                return (T) PyLong_AsUnsignedLong(m_ptr);
        } else {
            if (std::is_signed<T>::value)
                return (T) detail::PyLong_AsLongLong(m_ptr);
            else
                return (T) detail::PyLong_AsUnsignedLongLong(m_ptr);
        }
    }
};

class float_ : public object {
public:
    PYBIND11_OBJECT_DEFAULT(float_, object, PyFloat_Check)
    float_(float value) : object(PyFloat_FromDouble((double) value), false) { }
    float_(double value) : object(PyFloat_FromDouble((double) value), false) { }
    operator float() const { return (float) PyFloat_AsDouble(m_ptr); }
    operator double() const { return (double) PyFloat_AsDouble(m_ptr); }
};

class slice : public object {
public:
    PYBIND11_OBJECT_DEFAULT(slice, object, PySlice_Check)
    slice(ssize_t start_, ssize_t stop_, ssize_t step_) {
        int_ start(start_), stop(stop_), step(step_);
        m_ptr = PySlice_New(start.ptr(), stop.ptr(), step.ptr());
    }
    bool compute(ssize_t length, ssize_t *start, ssize_t *stop, ssize_t *step, ssize_t *slicelength) const {
        return PySlice_GetIndicesEx(
#if PY_MAJOR_VERSION >= 3
                m_ptr,
#else
                (PySliceObject *) m_ptr,
#endif
                length, start, stop, step, slicelength) == 0;
    }
};

class capsule : public object {
public:
    PYBIND11_OBJECT_DEFAULT(capsule, object, PyCapsule_CheckExact)
    capsule(PyObject *obj, bool borrowed) : object(obj, borrowed) { }
    capsule(void *value, void (*destruct)(PyObject *) = nullptr) : object(PyCapsule_New(value, nullptr, destruct), false) { }
    template <typename T> operator T *() const {
        T * result = static_cast<T *>(PyCapsule_GetPointer(m_ptr, nullptr));
        if (!result) throw std::runtime_error("Unable to extract capsule contents!");
        return result;
    }
};

class tuple : public object {
public:
    PYBIND11_OBJECT_DEFAULT(tuple, object, PyTuple_Check)
    tuple(size_t size) : object(PyTuple_New((Py_ssize_t) size), false) { }
    size_t size() const { return (size_t) PyTuple_Size(m_ptr); }
    detail::tuple_accessor operator[](size_t index) { return detail::tuple_accessor(ptr(), index); }
};

class dict : public object {
public:
    PYBIND11_OBJECT(dict, object, PyDict_Check)
    dict() : object(PyDict_New(), false) { }
    size_t size() const { return (size_t) PyDict_Size(m_ptr); }
    detail::dict_iterator begin() { return (++detail::dict_iterator(ptr(), 0)); }
    detail::dict_iterator end() { return detail::dict_iterator(); }
    void clear() { PyDict_Clear(ptr()); }
};

class list : public object {
public:
    PYBIND11_OBJECT(list, object, PyList_Check)
    list(size_t size = 0) : object(PyList_New((ssize_t) size), false) { }
    size_t size() const { return (size_t) PyList_Size(m_ptr); }
    detail::list_accessor operator[](size_t index) { return detail::list_accessor(ptr(), index); }
    void append(const object &object) { PyList_Append(m_ptr, (PyObject *) object.ptr()); }
};

class set : public object {
public:
    PYBIND11_OBJECT(set, object, PySet_Check)
    set() : object(PySet_New(nullptr), false) { }
    size_t size() const { return (size_t) PySet_Size(m_ptr); }
    void add(const object &object) { PySet_Add(m_ptr, (PyObject *) object.ptr()); }
    void clear() { PySet_Clear(ptr()); }
};

class function : public object {
public:
    PYBIND11_OBJECT_DEFAULT(function, object, PyFunction_Check)

    bool is_cpp_function() {
        PyObject *ptr = detail::get_function(m_ptr);
        return ptr != nullptr && PyCFunction_Check(ptr);
    }
};

class buffer : public object {
public:
    PYBIND11_OBJECT_DEFAULT(buffer, object, PyObject_CheckBuffer)

    buffer_info request(bool writable = false) {
        int flags = PyBUF_STRIDES | PyBUF_FORMAT;
        if (writable) flags |= PyBUF_WRITABLE;
        view = new Py_buffer();
        if (PyObject_GetBuffer(m_ptr, view, flags) != 0)
            throw error_already_set();
        std::vector<size_t> shape(view->ndim), strides(view->ndim);
        for (int i=0; i<view->ndim; ++i) {
            shape[i] = (size_t) view->shape[i];
            strides[i] = (size_t) view->strides[i];
        }
        return buffer_info(view->buf, view->itemsize, view->format,
                           view->ndim, shape, strides);
    }
    ~buffer() { if (view) { PyBuffer_Release(view); delete view; } }
private:
    Py_buffer *view = nullptr;
};

NAMESPACE_BEGIN(detail)
inline internals &get_internals() {
    static internals *internals_ptr = nullptr;
    if (internals_ptr)
        return *internals_ptr;
    handle builtins(PyEval_GetBuiltins());
    capsule caps(builtins["__pybind11__"]);
    if (caps.check()) {
        internals_ptr = caps;
    } else {
        internals_ptr = new internals();
        builtins["__pybind11__"] = capsule(internals_ptr);
    }
    return *internals_ptr;
}

inline std::string error_string() {
    std::string errorString;
    PyThreadState *tstate = PyThreadState_GET();
    if (tstate == nullptr)
        return "";

    if (tstate->curexc_type) {
        errorString += (const char *) handle(tstate->curexc_type).str();
        errorString += ": ";
    }
    if (tstate->curexc_value)
        errorString += (const char *) handle(tstate->curexc_value).str();

    return errorString;
}

inline handle get_object_handle(const void *ptr) {
    auto instances = get_internals().registered_instances;
    auto it = instances.find(ptr);
    if (it == instances.end())
        return handle();
    return it->second;
}

inline handle get_type_handle(const std::type_info &tp) {
    auto instances = get_internals().registered_types;
    auto it = instances.find(&tp);
    if (it == instances.end())
        return handle();
    return handle((PyObject *) it->second.type);
}

NAMESPACE_END(detail)
NAMESPACE_END(pybind11)
