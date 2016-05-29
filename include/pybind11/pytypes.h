/*
    pybind11/typeid.h: Convenience wrapper classes for basic Python types

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"
#include <utility>
#include <type_traits>

NAMESPACE_BEGIN(pybind11)

/* A few forward declarations */
class object; class str; class object; class dict; class iterator;
namespace detail { class accessor; class args_proxy; class kwargs_proxy; }

/// Holds a reference to a Python object (no reference counting)
class handle {
public:
    handle() : m_ptr(nullptr) { }
    handle(const handle &other) : m_ptr(other.m_ptr) { }
    handle(PyObject *ptr) : m_ptr(ptr) { }
    PyObject *ptr() const { return m_ptr; }
    PyObject *&ptr() { return m_ptr; }
    const handle& inc_ref() const { Py_XINCREF(m_ptr); return *this; }
    const handle& dec_ref() const { Py_XDECREF(m_ptr); return *this; }
    int ref_count() const { return (int) Py_REFCNT(m_ptr); }
    handle get_type() const { return handle((PyObject *) Py_TYPE(m_ptr)); }
    inline iterator begin() const;
    inline iterator end() const;
    inline detail::accessor operator[](handle key) const;
    inline detail::accessor operator[](const char *key) const;
    inline detail::accessor attr(handle key) const;
    inline detail::accessor attr(const char *key) const;
    inline pybind11::str str() const;
    template <typename T> T cast() const;
    template <typename ... Args>
    [[deprecated("call(...) was deprecated in favor of operator()(...)")]]
    object call(Args&&... args) const;
    template <typename ... Args> object operator()(Args&&... args) const;
    inline object operator()(detail::args_proxy args) const;
    inline object operator()(detail::args_proxy f_args, detail::kwargs_proxy kwargs) const;
    operator bool() const { return m_ptr != nullptr; }
    bool operator==(const handle &h) const { return m_ptr == h.m_ptr; }
    bool operator!=(const handle &h) const { return m_ptr != h.m_ptr; }
    bool check() const { return m_ptr != nullptr; }
    inline detail::args_proxy operator*() const;
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
    object(object &&other) noexcept { m_ptr = other.m_ptr; other.m_ptr = nullptr; }
    ~object() { dec_ref(); }

    handle release() {
      PyObject *tmp = m_ptr;
      m_ptr = nullptr;
      return handle(tmp);
    }

    object& operator=(const object &other) {
        other.inc_ref();
        dec_ref();
        m_ptr = other.m_ptr;
        return *this;
    }

    object& operator=(object &&other) noexcept {
        if (this != &other) {
            handle temp(m_ptr);
            m_ptr = other.m_ptr;
            other.m_ptr = nullptr;
            temp.dec_ref();
        }
        return *this;
    }
};

NAMESPACE_BEGIN(detail)
inline handle get_function(handle value) {
    if (value) {
#if PY_MAJOR_VERSION >= 3
        if (PyInstanceMethod_Check(value.ptr()))
            value = PyInstanceMethod_GET_FUNCTION(value.ptr());
#endif
        if (PyMethod_Check(value.ptr()))
            value = PyMethod_GET_FUNCTION(value.ptr());
    }
    return value;
}

class accessor {
public:
    accessor(handle obj, handle key, bool attr)
        : obj(obj), key(key, true), attr(attr) { }
    accessor(handle obj, const char *key, bool attr)
        : obj(obj), key(PyUnicode_FromString(key), false), attr(attr) { }
    accessor(const accessor &a) : obj(a.obj), key(a.key), attr(a.attr) { }

    void operator=(accessor o) { operator=(object(o)); }

    void operator=(const handle &value) {
        if (attr) {
            if (PyObject_SetAttr(obj.ptr(), key.ptr(), value.ptr()) == -1)
                pybind11_fail("Unable to set object attribute");
        } else {
            if (PyObject_SetItem(obj.ptr(), key.ptr(), value.ptr()) == -1)
                pybind11_fail("Unable to set object item");
        }
    }

    operator object() const {
        object result(attr ? PyObject_GetAttr(obj.ptr(), key.ptr())
                           : PyObject_GetItem(obj.ptr(), key.ptr()), false);
        if (!result) {PyErr_Clear(); }
        return result;
    }

    template <typename T> T cast() const { return operator object().cast<T>(); }

    operator bool() const {
        if (attr) {
            return (bool) PyObject_HasAttr(obj.ptr(), key.ptr());
        } else {
            object result(PyObject_GetItem(obj.ptr(), key.ptr()), false);
            if (!result) PyErr_Clear();
            return (bool) result;
        }
    };

private:
    handle obj;
    object key;
    bool attr;
};

struct list_accessor {
public:
    list_accessor(handle list, size_t index) : list(list), index(index) { }

    void operator=(list_accessor o) { return operator=(object(o)); }

    void operator=(const handle &o) {
        // PyList_SetItem steals a reference to 'o'
        if (PyList_SetItem(list.ptr(), (ssize_t) index, o.inc_ref().ptr()) < 0)
            pybind11_fail("Unable to assign value in Python list!");
    }

    operator object() const {
        PyObject *result = PyList_GetItem(list.ptr(), (ssize_t) index);
        if (!result)
            pybind11_fail("Unable to retrieve value from Python list!");
        return object(result, true);
    }

    template <typename T> T cast() const { return operator object().cast<T>(); }
private:
    handle list;
    size_t index;
};

struct tuple_accessor {
public:
    tuple_accessor(handle tuple, size_t index) : tuple(tuple), index(index) { }

    void operator=(tuple_accessor o) { return operator=(object(o)); }

    void operator=(const handle &o) {
        // PyTuple_SetItem steals a referenceto 'o'
        if (PyTuple_SetItem(tuple.ptr(), (ssize_t) index, o.inc_ref().ptr()) < 0)
            pybind11_fail("Unable to assign value in Python tuple!");
    }

    operator object() const {
        PyObject *result = PyTuple_GetItem(tuple.ptr(), (ssize_t) index);
        if (!result)
            pybind11_fail("Unable to retrieve value from Python tuple!");
        return object(result, true);
    }

    template <typename T> T cast() const { return operator object().cast<T>(); }
private:
    handle tuple;
    size_t index;
};

struct dict_iterator {
public:
    dict_iterator(handle dict = handle(), ssize_t pos = -1) : dict(dict), pos(pos) { }
    dict_iterator& operator++() {
        if (!PyDict_Next(dict.ptr(), &pos, &key.ptr(), &value.ptr()))
            pos = -1;
        return *this;
    }
    std::pair<handle, handle> operator*() const {
        return std::make_pair(key, value);
    }
    bool operator==(const dict_iterator &it) const { return it.pos == pos; }
    bool operator!=(const dict_iterator &it) const { return it.pos != pos; }
private:
    handle dict, key, value;
    ssize_t pos = 0;
};

inline bool PyIterable_Check(PyObject *obj) {
    PyObject *iter = PyObject_GetIter(obj);
    if (iter) {
        Py_DECREF(iter);
        return true;
    } else {
        PyErr_Clear();
        return false;
    }
}

inline bool PyNone_Check(PyObject *o) { return o == Py_None; }

inline bool PyUnicode_Check_Permissive(PyObject *o) { return PyUnicode_Check(o) || PYBIND11_BYTES_CHECK(o); }

class kwargs_proxy : public handle {
public:
    kwargs_proxy(handle h) : handle(h) { }
};

class args_proxy : public handle {
public:
    args_proxy(handle h) : handle(h) { }
    kwargs_proxy operator*() const { return kwargs_proxy(*this); }
};

NAMESPACE_END(detail)

#define PYBIND11_OBJECT_CVT(Name, Parent, CheckFun, CvtStmt) \
    public: \
        Name(const handle &h, bool borrowed) : Parent(h, borrowed) { CvtStmt; } \
        Name(const object& o): Parent(o) { CvtStmt; } \
        Name(object&& o) noexcept : Parent(std::move(o)) { CvtStmt; } \
        Name& operator=(object&& o) noexcept { (void) object::operator=(std::move(o)); CvtStmt; return *this; } \
        Name& operator=(const object& o) { return static_cast<Name&>(object::operator=(o)); CvtStmt; } \
        bool check() const { return m_ptr != nullptr && (bool) CheckFun(m_ptr); }

#define PYBIND11_OBJECT(Name, Parent, CheckFun) \
    PYBIND11_OBJECT_CVT(Name, Parent, CheckFun, )

#define PYBIND11_OBJECT_DEFAULT(Name, Parent, CheckFun) \
    PYBIND11_OBJECT(Name, Parent, CheckFun) \
    Name() : Parent() { }

class iterator : public object {
public:
    PYBIND11_OBJECT_CVT(iterator, object, PyIter_Check, value = object(); ready = false)
    iterator() : object(), value(object()), ready(false) { }
    iterator(const iterator& it) : object(it), value(it.value), ready(it.ready) { }
    iterator(iterator&& it) : object(std::move(it)), value(std::move(it.value)), ready(it.ready) { }

    /** Caveat: this copy constructor does not (and cannot) clone the internal
        state of the Python iterable */
    iterator &operator=(const iterator &it) {
        (void) object::operator=(it);
        value = it.value;
        ready = it.ready;
        return *this;
    }

    iterator &operator=(iterator &&it) noexcept {
        (void) object::operator=(std::move(it));
        value = std::move(it.value);
        ready = it.ready;
        return *this;
    }

    iterator& operator++() {
        if (m_ptr)
            advance();
        return *this;
    }

    /** Caveat: this postincrement operator does not (and cannot) clone the
        internal state of the Python iterable. It should only be used to
        retrieve the current iterate using <tt>operator*()</tt> */
    iterator operator++(int) {
        iterator rv(*this);
        rv.value = value;
        if (m_ptr)
            advance();
        return rv;
    }

    bool operator==(const iterator &it) const { return *it == **this; }
    bool operator!=(const iterator &it) const { return *it != **this; }

    handle operator*() const {
        if (!ready && m_ptr) {
            auto& self = const_cast<iterator &>(*this);
            self.advance();
            self.ready = true;
        }
        return value;
    }

private:
    void advance() { value = object(PyIter_Next(m_ptr), false); }

private:
    object value;
    bool ready;
};

class iterable : public object {
public:
    PYBIND11_OBJECT_DEFAULT(iterable, object, detail::PyIterable_Check)
};

inline detail::accessor handle::operator[](handle key) const { return detail::accessor(*this, key, false); }
inline detail::accessor handle::operator[](const char *key) const { return detail::accessor(*this, key, false); }
inline detail::accessor handle::attr(handle key) const { return detail::accessor(*this, key, true); }
inline detail::accessor handle::attr(const char *key) const { return detail::accessor(*this, key, true); }
inline iterator handle::begin() const { return iterator(PyObject_GetIter(ptr()), false); }
inline iterator handle::end() const { return iterator(nullptr, false); }
inline detail::args_proxy handle::operator*() const { return detail::args_proxy(*this); }

class str : public object {
public:
    PYBIND11_OBJECT_DEFAULT(str, object, detail::PyUnicode_Check_Permissive)

    str(const std::string &s)
        : object(PyUnicode_FromStringAndSize(s.c_str(), (ssize_t) s.length()), false) {
        if (!m_ptr) pybind11_fail("Could not allocate string object!");
    }

    operator std::string() const {
        object temp = *this;
        if (PyUnicode_Check(m_ptr)) {
            temp = object(PyUnicode_AsUTF8String(m_ptr), false);
            if (!temp)
                pybind11_fail("Unable to extract string contents! (encoding issue)");
        }
        char *buffer;
        ssize_t length;
        int err = PYBIND11_BYTES_AS_STRING_AND_SIZE(temp.ptr(), &buffer, &length);
        if (err == -1)
            pybind11_fail("Unable to extract string contents! (invalid type)");
        return std::string(buffer, (size_t) length);
    }
};

inline pybind11::str handle::str() const {
    PyObject *strValue = PyObject_Str(m_ptr);
#if PY_MAJOR_VERSION < 3
    PyObject *unicode = PyUnicode_FromEncodedObject(strValue, "utf-8", nullptr);
    Py_XDECREF(strValue); strValue = unicode;
#endif
    return pybind11::str(strValue, false);
}

class bytes : public object {
public:
    PYBIND11_OBJECT_DEFAULT(bytes, object, PYBIND11_BYTES_CHECK)

    bytes(const std::string &s)
        : object(PYBIND11_BYTES_FROM_STRING_AND_SIZE(s.data(), (ssize_t) s.size()), false) {
        if (!m_ptr) pybind11_fail("Could not allocate bytes object!");
    }

    operator std::string() const {
        char *buffer;
        ssize_t length;
        int err = PYBIND11_BYTES_AS_STRING_AND_SIZE(m_ptr, &buffer, &length);
        if (err == -1)
            pybind11_fail("Unable to extract bytes contents!");
        return std::string(buffer, (size_t) length);
    }
};

class none : public object {
public:
    PYBIND11_OBJECT(none, object, detail::PyNone_Check)
    none() : object(Py_None, true) { }
};

class bool_ : public object {
public:
    PYBIND11_OBJECT_DEFAULT(bool_, object, PyBool_Check)
    bool_(bool value) : object(value ? Py_True : Py_False, true) { }
    operator bool() const { return m_ptr && PyLong_AsLong(m_ptr) != 0; }
};

class int_ : public object {
public:
    PYBIND11_OBJECT_DEFAULT(int_, object, PYBIND11_LONG_CHECK)
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
        if (!m_ptr) pybind11_fail("Could not allocate int object!");
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
                return (T) PYBIND11_LONG_AS_LONGLONG(m_ptr);
            else
                return (T) PYBIND11_LONG_AS_UNSIGNED_LONGLONG(m_ptr);
        }
    }
};

class float_ : public object {
public:
    PYBIND11_OBJECT_DEFAULT(float_, object, PyFloat_Check)
    float_(float value) : object(PyFloat_FromDouble((double) value), false) {
        if (!m_ptr) pybind11_fail("Could not allocate float object!");
    }
    float_(double value) : object(PyFloat_FromDouble((double) value), false) {
        if (!m_ptr) pybind11_fail("Could not allocate float object!");
    }
    operator float() const { return (float) PyFloat_AsDouble(m_ptr); }
    operator double() const { return (double) PyFloat_AsDouble(m_ptr); }
};

class weakref : public object {
public:
    PYBIND11_OBJECT_DEFAULT(weakref, object, PyWeakref_Check)
    weakref(handle obj, handle callback = handle()) : object(PyWeakref_NewRef(obj.ptr(), callback.ptr()), false) {
        if (!m_ptr) pybind11_fail("Could not allocate weak reference!");
    }
};

class slice : public object {
public:
    PYBIND11_OBJECT_DEFAULT(slice, object, PySlice_Check)
    slice(ssize_t start_, ssize_t stop_, ssize_t step_) {
        int_ start(start_), stop(stop_), step(step_);
        m_ptr = PySlice_New(start.ptr(), stop.ptr(), step.ptr());
        if (!m_ptr) pybind11_fail("Could not allocate slice object!");
    }
    bool compute(size_t length, size_t *start, size_t *stop, size_t *step,
                 size_t *slicelength) const {
        return PySlice_GetIndicesEx((PYBIND11_SLICE_OBJECT *) m_ptr,
                                    (ssize_t) length, (ssize_t *) start,
                                    (ssize_t *) stop, (ssize_t *) step,
                                    (ssize_t *) slicelength) == 0;
    }
};

class capsule : public object {
public:
    PYBIND11_OBJECT_DEFAULT(capsule, object, PyCapsule_CheckExact)
    capsule(PyObject *obj, bool borrowed) : object(obj, borrowed) { }
    capsule(const void *value, void (*destruct)(PyObject *) = nullptr)
        : object(PyCapsule_New(const_cast<void*>(value), nullptr, destruct), false) {
        if (!m_ptr) pybind11_fail("Could not allocate capsule object!");
    }
    template <typename T> operator T *() const {
        T * result = static_cast<T *>(PyCapsule_GetPointer(m_ptr, nullptr));
        if (!result) pybind11_fail("Unable to extract capsule contents!");
        return result;
    }
};

class tuple : public object {
public:
    PYBIND11_OBJECT(tuple, object, PyTuple_Check)
    tuple(size_t size = 0) : object(PyTuple_New((ssize_t) size), false) {
        if (!m_ptr) pybind11_fail("Could not allocate tuple object!");
    }
    size_t size() const { return (size_t) PyTuple_Size(m_ptr); }
    detail::tuple_accessor operator[](size_t index) const { return detail::tuple_accessor(*this, index); }
};

class dict : public object {
public:
    PYBIND11_OBJECT(dict, object, PyDict_Check)
    dict() : object(PyDict_New(), false) {
        if (!m_ptr) pybind11_fail("Could not allocate dict object!");
    }
    size_t size() const { return (size_t) PyDict_Size(m_ptr); }
    detail::dict_iterator begin() const { return (++detail::dict_iterator(*this, 0)); }
    detail::dict_iterator end() const { return detail::dict_iterator(); }
    void clear() const { PyDict_Clear(ptr()); }
};

class list : public object {
public:
    PYBIND11_OBJECT(list, object, PyList_Check)
    list(size_t size = 0) : object(PyList_New((ssize_t) size), false) {
        if (!m_ptr) pybind11_fail("Could not allocate list object!");
    }
    size_t size() const { return (size_t) PyList_Size(m_ptr); }
    detail::list_accessor operator[](size_t index) const { return detail::list_accessor(*this, index); }
    void append(const object &object) const { PyList_Append(m_ptr, object.ptr()); }
};

class args : public tuple { PYBIND11_OBJECT_DEFAULT(args, tuple, PyTuple_Check) };
class kwargs : public dict { PYBIND11_OBJECT_DEFAULT(kwargs, dict, PyDict_Check)  };

class set : public object {
public:
    PYBIND11_OBJECT(set, object, PySet_Check)
    set() : object(PySet_New(nullptr), false) {
        if (!m_ptr) pybind11_fail("Could not allocate set object!");
    }
    size_t size() const { return (size_t) PySet_Size(m_ptr); }
    bool add(const object &object) const { return PySet_Add(m_ptr, object.ptr()) == 0; }
    void clear() const { PySet_Clear(m_ptr); }
};

class function : public object {
public:
    PYBIND11_OBJECT_DEFAULT(function, object, PyCallable_Check)
    bool is_cpp_function() const {
        handle fun = detail::get_function(m_ptr);
        return fun && PyCFunction_Check(fun.ptr());
    }
};

class buffer : public object {
public:
    PYBIND11_OBJECT_DEFAULT(buffer, object, PyObject_CheckBuffer)

    buffer_info request(bool writable = false) {
        int flags = PyBUF_STRIDES | PyBUF_FORMAT;
        if (writable) flags |= PyBUF_WRITABLE;
        Py_buffer *view = new Py_buffer();
        if (PyObject_GetBuffer(m_ptr, view, flags) != 0)
            throw error_already_set();
        return buffer_info(view);
    }
};

inline size_t len(handle h) {
    ssize_t result = PyObject_Length(h.ptr());
    if (result < 0)
        pybind11_fail("Unable to compute length of object");
    return (size_t) result;
}

NAMESPACE_END(pybind11)
