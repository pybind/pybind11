/*
    pybind11/stl.h: Transparent conversion for STL data types

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "pybind11.h"
#include "utility.h"
#include <set>
#include <unordered_set>
#include <map>
#include <unordered_map>
#include <iostream>
#include <list>
#include <valarray>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#endif

NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
NAMESPACE_BEGIN(detail)

/// Extracts an const lvalue reference or rvalue reference for U based on the type of T (e.g. for
/// forwarding a container element).  Typically used indirect via forwarded_type(), below.
template <typename T, typename U>
using forwarded_type = conditional_t<
    std::is_lvalue_reference<T>::value, remove_reference_t<U> &, remove_reference_t<U> &&>;

/// Forwards a value U as rvalue or lvalue according to whether T is rvalue or lvalue; typically
/// used for forwarding a container's elements.
template <typename T, typename U>
forwarded_type<T, U> forward_like(U &&u) {
    return std::forward<detail::forwarded_type<T, U>>(std::forward<U>(u));
}

template <typename Type, typename Key> struct set_caster {
    using type = Type;
    using key_conv = make_caster<Key>;

    bool load(handle src, bool convert) {
        if (!isinstance<pybind11::set>(src))
            return false;
        auto s = reinterpret_borrow<pybind11::set>(src);
        value.clear();
        for (auto entry : s) {
            key_conv conv;
            if (!conv.load(entry, convert))
                return false;
            value.insert(cast_op<Key &&>(std::move(conv)));
        }
        return true;
    }

    template <typename T>
    static handle cast(T &&src, return_value_policy policy, handle parent) {
        pybind11::set s;
        for (auto &&value : src) {
            auto value_ = reinterpret_steal<object>(key_conv::cast(forward_like<T>(value), policy, parent));
            if (!value_ || !s.add(value_))
                return handle();
        }
        return s.release();
    }

    PYBIND11_TYPE_CASTER(type, _("Set[") + key_conv::name + _("]"));
};

template <typename Type, typename Key, typename Value> struct map_caster {
    using key_conv   = make_caster<Key>;
    using value_conv = make_caster<Value>;

    bool load(handle src, bool convert) {
        if (!isinstance<dict>(src))
            return false;
        auto d = reinterpret_borrow<dict>(src);
        value.clear();
        for (auto it : d) {
            key_conv kconv;
            value_conv vconv;
            if (!kconv.load(it.first.ptr(), convert) ||
                !vconv.load(it.second.ptr(), convert))
                return false;
            value.emplace(cast_op<Key &&>(std::move(kconv)), cast_op<Value &&>(std::move(vconv)));
        }
        return true;
    }

    template <typename T>
    static handle cast(T &&src, return_value_policy policy, handle parent) {
        dict d;
        for (auto &&kv : src) {
            auto key = reinterpret_steal<object>(key_conv::cast(forward_like<T>(kv.first), policy, parent));
            auto value = reinterpret_steal<object>(value_conv::cast(forward_like<T>(kv.second), policy, parent));
            if (!key || !value)
                return handle();
            d[key] = value;
        }
        return d.release();
    }

    PYBIND11_TYPE_CASTER(Type, _("Dict[") + key_conv::name + _(", ") + value_conv::name + _("]"));
};

template <typename Type, typename Value> struct list_caster {
    using value_conv = make_caster<Value>;

    bool load(handle src, bool convert) {
        if (!isinstance<sequence>(src))
            return false;
        auto s = reinterpret_borrow<sequence>(src);
        value.clear();
        reserve_maybe(s, &value);
        for (auto it : s) {
            value_conv conv;
            if (!conv.load(it, convert))
                return false;
            value.push_back(cast_op<Value &&>(std::move(conv)));
        }
        return true;
    }

private:
    template <typename T = Type,
              enable_if_t<std::is_same<decltype(std::declval<T>().reserve(0)), void>::value, int> = 0>
    void reserve_maybe(sequence s, Type *) { value.reserve(s.size()); }
    void reserve_maybe(sequence, void *) { }

public:
    template <typename T>
    static handle cast(T &&src, return_value_policy policy, handle parent) {
        list l(src.size());
        size_t index = 0;
        for (auto &&value : src) {
            auto value_ = reinterpret_steal<object>(value_conv::cast(forward_like<T>(value), policy, parent));
            if (!value_)
                return handle();
            PyList_SET_ITEM(l.ptr(), (ssize_t) index++, value_.release().ptr()); // steals a reference
        }
        return l.release();
    }

    PYBIND11_TYPE_CASTER(Type, _("List[") + value_conv::name + _("]"));
};

template <typename Type, typename Alloc> struct type_caster<std::vector<Type, Alloc>>
 : list_caster<std::vector<Type, Alloc>, Type> { };

template <typename Type, typename Alloc> struct type_caster<std::list<Type, Alloc>>
 : list_caster<std::list<Type, Alloc>, Type> { };

template <typename ArrayType, typename Value, bool Resizable, size_t Size = 0> struct array_caster {
    using value_conv = make_caster<Value>;

private:
    template <bool R = Resizable>
    bool require_size(enable_if_t<R, size_t> size) {
        if (value.size() != size)
            value.resize(size);
        return true;
    }
    template <bool R = Resizable>
    bool require_size(enable_if_t<!R, size_t> size) {
        return size == Size;
    }

public:
    bool load(handle src, bool convert) {
        if (!isinstance<list>(src))
            return false;
        auto l = reinterpret_borrow<list>(src);
        if (!require_size(l.size()))
            return false;
        size_t ctr = 0;
        for (auto it : l) {
            value_conv conv;
            if (!conv.load(it, convert))
                return false;
            value[ctr++] = cast_op<Value &&>(std::move(conv));
        }
        return true;
    }

    template <typename T>
    static handle cast(T &&src, return_value_policy policy, handle parent) {
        list l(src.size());
        size_t index = 0;
        for (auto &&value : src) {
            auto value_ = reinterpret_steal<object>(value_conv::cast(forward_like<T>(value), policy, parent));
            if (!value_)
                return handle();
            PyList_SET_ITEM(l.ptr(), (ssize_t) index++, value_.release().ptr()); // steals a reference
        }
        return l.release();
    }

    PYBIND11_TYPE_CASTER(ArrayType, _("List[") + value_conv::name + _<Resizable>(_(""), _("[") + _<Size>() + _("]")) + _("]"));
};

template <typename Type, size_t Size> struct type_caster<std::array<Type, Size>>
 : array_caster<std::array<Type, Size>, Type, false, Size> { };

template <typename Type> struct type_caster<std::valarray<Type>>
 : array_caster<std::valarray<Type>, Type, true> { };

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
    os << (std::string) str(obj);
    return os;
}

NAMESPACE_END(PYBIND11_NAMESPACE)

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
