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
    bool load(handle src, bool convert) {
        list l(src, true);
        if (!l.check())
            return false;
        value.reserve(l.size());
        value.clear();
        value_conv conv;
        for (auto it : l) {
            if (!conv.load(it, convert))
                return false;
            value.push_back((Value) conv);
        }
        return true;
    }

    static handle cast(const type &src, return_value_policy policy, handle parent) {
        list l(src.size());
        size_t index = 0;
        for (auto const &value: src) {
            object value_ = object(value_conv::cast(value, policy, parent), false);
            if (!value_)
                return handle();
            PyList_SET_ITEM(l.ptr(), index++, value_.release().ptr()); // steals a reference
        }
        return l.release();
    }
    PYBIND11_TYPE_CASTER(type, _("list<") + value_conv::name() + _(">"));
};

template <typename Key, typename Compare, typename Alloc> struct type_caster<std::set<Key, Compare, Alloc>> {
    typedef std::set<Key, Compare, Alloc> type;
    typedef type_caster<Key> key_conv;
public:
    bool load(handle src, bool convert) {
        pybind11::set s(src, true);
        if (!s.check())
            return false;
        value.clear();
        key_conv conv;
        for (auto entry : s) {
            if (!conv.load(entry, convert))
                return false;
            value.insert((Key) conv);
        }
        return true;
    }

    static handle cast(const type &src, return_value_policy policy, handle parent) {
        pybind11::set s;
        for (auto const &value: src) {
            object value_ = object(key_conv::cast(value, policy, parent), false);
            if (!value_ || !s.add(value_))
                return handle();
        }
        return s.release();
    }
    PYBIND11_TYPE_CASTER(type, _("set<") + key_conv::name() + _(">"));
};

template <typename Key, typename Value, typename Compare, typename Alloc> struct type_caster<std::map<Key, Value, Compare, Alloc>> {
public:
    typedef std::map<Key, Value, Compare, Alloc>  type;
    typedef type_caster<Key>   key_conv;
    typedef type_caster<Value> value_conv;

    bool load(handle src, bool convert) {
        dict d(src, true);
        if (!d.check())
            return false;
        key_conv kconv;
        value_conv vconv;
        value.clear();
        for (auto it : d) {
            if (!kconv.load(it.first.ptr(), convert) ||
                !vconv.load(it.second.ptr(), convert))
                return false;
            value[(Key) kconv] = (Value) vconv;
        }
        return true;
    }

    static handle cast(const type &src, return_value_policy policy, handle parent) {
        dict d;
        for (auto const &kv: src) {
            object key = object(key_conv::cast(kv.first, policy, parent), false);
            object value = object(value_conv::cast(kv.second, policy, parent), false);
            if (!key || !value)
                return handle();
            d[key] = value;
        }
        return d.release();
    }

    PYBIND11_TYPE_CASTER(type, _("dict<") + key_conv::name() + _(", ") + value_conv::name() + _(">"));
};

NAMESPACE_END(detail)

inline std::ostream &operator<<(std::ostream &os, const handle &obj) {
    os << (std::string) obj.str();
    return os;
}

NAMESPACE_END(pybind11)

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
