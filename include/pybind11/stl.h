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
        object list(PyList_New(src.size()), false);
        if (!list)
            return nullptr;
        size_t index = 0;
        for (auto const &value: src) {
            object value_ (value_conv::cast(value, policy, parent), false);
            if (!value_)
                return nullptr;
            PyList_SET_ITEM(list.ptr(), index++, value_.release()); // steals a reference
        }
        return list.release();
    }
    PYBIND11_TYPE_CASTER(type, _("list<") + value_conv::name() + _(">"));
};

template <typename Key, typename Compare, typename Alloc> struct type_caster<std::set<Key, Compare, Alloc>> {
    typedef std::set<Key, Compare, Alloc> type;
    typedef type_caster<Key> key_conv;
public:
    bool load(PyObject *src, bool convert) {
        pybind11::set s(src, true);
        if (!s.check())
            return false;
        value.clear();
        key_conv conv;
        for (const object &o: s) {
            if (!conv.load((PyObject *) o.ptr(), convert))
                return false;
            value.insert((Key) conv);
        }
        return true;
    }

    static PyObject *cast(const type &src, return_value_policy policy, PyObject *parent) {
        object set(PySet_New(nullptr), false);
        if (!set)
            return nullptr;
        for (auto const &value: src) {
            object value_(key_conv::cast(value, policy, parent), false);
            if (!value_ || PySet_Add(set.ptr(), value_.ptr()) != 0)
                return nullptr;
        }
        return set.release();
    }
    PYBIND11_TYPE_CASTER(type, _("set<") + key_conv::name() + _(">"));
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
        object dict(PyDict_New(), false);
        if (!dict)
            return nullptr;
        for (auto const &kv: src) {
            object key(key_conv::cast(kv.first, policy, parent), false);
            object value(value_conv::cast(kv.second, policy, parent), false);
            if (!key || !value || PyDict_SetItem(dict.ptr(), key.ptr(), value.ptr()) != 0)
                return nullptr;
        }
        return dict.release();
    }

    PYBIND11_TYPE_CASTER(type, _("dict<") + key_conv::name() + _(", ") + value_conv::name() + _(">"));
};

NAMESPACE_END(detail)

inline std::ostream &operator<<(std::ostream &os, const object &obj) { os << (const char *) obj.str(); return os; }

NAMESPACE_END(pybind11)

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
