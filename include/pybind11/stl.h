/*
    pybind11/stl.h: Transparent conversion for STL data types

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "pybind11.h"
#include <set>
#include <unordered_set>
#include <map>
#include <unordered_map>
#include <iostream>
#include <list>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#endif

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

template <typename Type, typename Key> struct set_caster {
    typedef Type type;
    typedef type_caster<typename intrinsic_type<Key>::type> key_conv;

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

template <typename Type, typename Key, typename Value> struct map_caster {
    typedef Type type;
    typedef type_caster<typename intrinsic_type<Key>::type>   key_conv;
    typedef type_caster<typename intrinsic_type<Value>::type> value_conv;

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
            value.emplace((Key) kconv, (Value) vconv);
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

template <typename Type, typename Value> struct list_caster {
    typedef Type type;
    typedef type_caster<typename intrinsic_type<Value>::type> value_conv;

    bool load(handle src, bool convert) {
        list l(src, true);
        if (!l.check())
            return false;
        value_conv conv;
        value.clear();
        reserve_maybe(l, &value);
        for (auto it : l) {
            if (!conv.load(it, convert))
                return false;
            value.push_back((Value) conv);
        }
        return true;
    }

    template <typename T = Type,
              typename std::enable_if<std::is_same<decltype(std::declval<T>().reserve(0)), void>::value, int>::type = 0>
    void reserve_maybe(list l, Type *) { value.reserve(l.size()); }
    void reserve_maybe(list, void *) { }

    static handle cast(const Type &src, return_value_policy policy, handle parent) {
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

    PYBIND11_TYPE_CASTER(Type, _("list<") + value_conv::name() + _(">"));
};

template <typename Type, typename Alloc> struct type_caster<std::vector<Type, Alloc>>
 : list_caster<std::vector<Type, Alloc>, Type> { };

template <typename Type, typename Alloc> struct type_caster<std::list<Type, Alloc>>
 : list_caster<std::list<Type, Alloc>, Type> { };

template <typename Type, size_t Size> struct type_caster<std::array<Type, Size>> {
    typedef std::array<Type, Size> array_type;
    typedef type_caster<typename intrinsic_type<Type>::type> value_conv;

    bool load(handle src, bool convert) {
        list l(src, true);
        if (!l.check())
            return false;
        if (l.size() != Size)
            return false;
        value_conv conv;
        size_t ctr = 0;
        for (auto it : l) {
            if (!conv.load(it, convert))
                return false;
            value[ctr++] = (Type) conv;
        }
        return true;
    }

    static handle cast(const array_type &src, return_value_policy policy, handle parent) {
        list l(Size);
        size_t index = 0;
        for (auto const &value: src) {
            object value_ = object(value_conv::cast(value, policy, parent), false);
            if (!value_)
                return handle();
            PyList_SET_ITEM(l.ptr(), index++, value_.release().ptr()); // steals a reference
        }
        return l.release();
    }
    PYBIND11_TYPE_CASTER(array_type, _("list<") + value_conv::name() + _(">") + _("[") + _<Size>() + _("]"));
};

template <typename Key, typename Compare, typename Alloc> struct type_caster<std::set<Key, Compare, Alloc>>
  : set_caster<std::set<Key, Compare, Alloc>, Key> { };

template <typename Key, typename Hash, typename Equal, typename Alloc> struct type_caster<std::unordered_set<Key, Hash, Equal, Alloc>>
  : set_caster<std::unordered_set<Key, Hash, Equal, Alloc>, Key> { };

template <typename Key, typename Value, typename Compare, typename Alloc> struct type_caster<std::map<Key, Value, Compare, Alloc>>
  : map_caster<std::map<Key, Value, Compare, Alloc>, Key, Value> { };

template <typename Key, typename Value, typename Hash, typename Equal, typename Alloc> struct type_caster<std::unordered_map<Key, Value, Hash, Equal, Alloc>>
  : map_caster<std::unordered_map<Key, Value, Hash, Equal, Alloc>, Key, Value> { };

NAMESPACE_END(detail)

inline std::ostream &operator<<(std::ostream &os, const handle &obj) {
    os << (std::string) obj.str();
    return os;
}

NAMESPACE_END(pybind11)

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
