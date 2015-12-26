/*
    pybind11/complex.h: Complex number support

    Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "pybind11.h"
#include <map>
#include <set>
#include <iostream>


#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#endif

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

template <typename Value, typename Alloc> struct type_caster<std::vector<Value, Alloc>> {
    typedef std::vector<Value, Alloc> type;
    typedef type_caster<Value> value_conv;
public:
    bool load(PyObject *src, bool convert) {
        if (!PyList_Check(src))
            return false;
        size_t size = (size_t) PyList_GET_SIZE(src);
        value.reserve(size);
        value.clear();
        value_conv conv;
        for (size_t i=0; i<size; ++i) {
            if (!conv.load(PyList_GetItem(src, (ssize_t) i), convert))
                return false;
            value.push_back((Value) conv);
        }
        return true;
    }

    static PyObject *cast(const type &src, return_value_policy policy, PyObject *parent) {
        PyObject *list = PyList_New(src.size());
        size_t index = 0;
        for (auto const &value: src) {
            PyObject *value_ = value_conv::cast(value, policy, parent);
            if (!value_) {
                Py_DECREF(list);
                return nullptr;
            }
            PyList_SET_ITEM(list, index++, value_); // steals a reference
        }
        return list;
    }
    PYBIND11_TYPE_CASTER(type, detail::descr("list<") + value_conv::name() + detail::descr(">"));
};

template <typename Value, typename Compare, typename Alloc> struct type_caster<std::set<Value, Compare, Alloc>> {
    typedef std::set<Value, Compare, Alloc> type;
    typedef type_caster<Value> value_conv;
public:
    bool load(PyObject *src, bool convert) {
        pybind11::set s(src, true);
        if (!s.check())
            return false;
        value.clear();
        value_conv conv;
        for (const object &o: s) {
            if (!conv.load((PyObject *) o.ptr(), convert))
                return false;
            value.insert((Value) conv);
        }
        return true;
    }

    static PyObject *cast(const type &src, return_value_policy policy, PyObject *parent) {
        PyObject *set = PySet_New(nullptr);
        for (auto const &value: src) {
            PyObject *value_ = value_conv::cast(value, policy, parent);
            if (!value_ || PySet_Add(set, value_) != 0) {
                Py_XDECREF(value_);
                Py_DECREF(set);
                return nullptr;
            }
            Py_DECREF(value_);
        }
        return set;
    }
    PYBIND11_TYPE_CASTER(type, detail::descr("set<") + value_conv::name() + detail::descr(">"));
};

template <typename Key, typename Value, typename Compare, typename Alloc> struct type_caster<std::map<Key, Value, Compare, Alloc>> {
public:
    typedef std::map<Key, Value, Compare, Alloc>  type;
    typedef type_caster<Key>   key_conv;
    typedef type_caster<Value> value_conv;

    bool load(PyObject *src, bool convert) {
        if (!PyDict_Check(src))
            return false;

        value.clear();
        PyObject *key_, *value_;
        ssize_t pos = 0;
        key_conv kconv;
        value_conv vconv;
        while (PyDict_Next(src, &pos, &key_, &value_)) {
            if (!kconv.load(key_, convert) || !vconv.load(value_, convert))
                return false;
            value[(Key) kconv] = (Value) vconv;
        }
        return true;
    }

    static PyObject *cast(const type &src, return_value_policy policy, PyObject *parent) {
        PyObject *dict = PyDict_New();
        for (auto const &kv: src) {
            PyObject *key   = key_conv::cast(kv.first, policy, parent);
            PyObject *value = value_conv::cast(kv.second, policy, parent);
            if (!key || !value || PyDict_SetItem(dict, key, value) != 0) {
                Py_XDECREF(key);
                Py_XDECREF(value);
                Py_DECREF(dict);
                return nullptr;
            }
            Py_DECREF(key);
            Py_DECREF(value);
        }
        return dict;
    }

    PYBIND11_TYPE_CASTER(type, detail::descr("dict<") + key_conv::name() + detail::descr(", ") + value_conv::name() + detail::descr(">"));
};

NAMESPACE_END(detail)

inline std::ostream &operator<<(std::ostream &os, const object &obj) { os << (const char *) obj.str(); return os; }

NAMESPACE_END(pybind11)

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
