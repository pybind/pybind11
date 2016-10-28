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

#ifdef __has_include
// std::optional (but including it in c++14 mode isn't allowed)
#  if defined(PYBIND11_CPP17) && __has_include(<optional>)
#    include <optional>
#    define PYBIND11_HAS_OPTIONAL 1
#  endif
// std::experimental::optional (but not allowed in c++11 mode)
#  if defined(PYBIND11_CPP14) && __has_include(<experimental/optional>)
#    include <experimental/optional>
#    if __cpp_lib_experimental_optional  // just in case
#      define PYBIND11_HAS_EXP_OPTIONAL 1
#    endif
#  endif
#endif

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

template <typename Type, typename Key> struct set_caster {
    using type = Type;
    using key_conv = make_caster<Key>;

    bool load(handle src, bool convert) {
        if (!isinstance<pybind11::set>(src))
            return false;
        auto s = reinterpret_borrow<pybind11::set>(src);
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
            auto value_ = reinterpret_steal<object>(key_conv::cast(value, policy, parent));
            if (!value_ || !s.add(value_))
                return handle();
        }
        return s.release();
    }

    PYBIND11_TYPE_CASTER(type, _("Set[") + key_conv::name() + _("]"));
};

template <typename Type, typename Key, typename Value> struct map_caster {
    using type = Type;
    using key_conv   = make_caster<Key>;
    using value_conv = make_caster<Value>;

    bool load(handle src, bool convert) {
        if (!isinstance<dict>(src))
            return false;
        auto d = reinterpret_borrow<dict>(src);
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
            auto key = reinterpret_steal<object>(key_conv::cast(kv.first, policy, parent));
            auto value = reinterpret_steal<object>(value_conv::cast(kv.second, policy, parent));
            if (!key || !value)
                return handle();
            d[key] = value;
        }
        return d.release();
    }

    PYBIND11_TYPE_CASTER(type, _("Dict[") + key_conv::name() + _(", ") + value_conv::name() + _("]"));
};

template <typename Type, typename Value> struct list_caster {
    using type = Type;
    using value_conv = make_caster<Value>;

    bool load(handle src, bool convert) {
        if (!isinstance<sequence>(src))
            return false;
        auto s = reinterpret_borrow<sequence>(src);
        value_conv conv;
        value.clear();
        reserve_maybe(s, &value);
        for (auto it : s) {
            if (!conv.load(it, convert))
                return false;
            value.push_back((Value) conv);
        }
        return true;
    }

    template <typename T = Type,
              enable_if_t<std::is_same<decltype(std::declval<T>().reserve(0)), void>::value, int> = 0>
    void reserve_maybe(sequence s, Type *) { value.reserve(s.size()); }
    void reserve_maybe(sequence, void *) { }

    static handle cast(const Type &src, return_value_policy policy, handle parent) {
        list l(src.size());
        size_t index = 0;
        for (auto const &value: src) {
            auto value_ = reinterpret_steal<object>(value_conv::cast(value, policy, parent));
            if (!value_)
                return handle();
            PyList_SET_ITEM(l.ptr(), index++, value_.release().ptr()); // steals a reference
        }
        return l.release();
    }

    PYBIND11_TYPE_CASTER(Type, _("List[") + value_conv::name() + _("]"));
};

template <typename Type, typename Alloc> struct type_caster<std::vector<Type, Alloc>>
 : list_caster<std::vector<Type, Alloc>, Type> { };

template <typename Type, typename Alloc> struct type_caster<std::list<Type, Alloc>>
 : list_caster<std::list<Type, Alloc>, Type> { };

template <typename Type, size_t Size> struct type_caster<std::array<Type, Size>> {
    using array_type = std::array<Type, Size>;
    using value_conv = make_caster<Type>;

    bool load(handle src, bool convert) {
        if (!isinstance<list>(src))
            return false;
        auto l = reinterpret_borrow<list>(src);
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
            auto value_ = reinterpret_steal<object>(value_conv::cast(value, policy, parent));
            if (!value_)
                return handle();
            PyList_SET_ITEM(l.ptr(), index++, value_.release().ptr()); // steals a reference
        }
        return l.release();
    }
    PYBIND11_TYPE_CASTER(array_type, _("List[") + value_conv::name() + _("[") + _<Size>() + _("]]"));
};

template <typename Key, typename Compare, typename Alloc> struct type_caster<std::set<Key, Compare, Alloc>>
  : set_caster<std::set<Key, Compare, Alloc>, Key> { };

template <typename Key, typename Hash, typename Equal, typename Alloc> struct type_caster<std::unordered_set<Key, Hash, Equal, Alloc>>
  : set_caster<std::unordered_set<Key, Hash, Equal, Alloc>, Key> { };

template <typename Key, typename Value, typename Compare, typename Alloc> struct type_caster<std::map<Key, Value, Compare, Alloc>>
  : map_caster<std::map<Key, Value, Compare, Alloc>, Key, Value> { };

template <typename Key, typename Value, typename Hash, typename Equal, typename Alloc> struct type_caster<std::unordered_map<Key, Value, Hash, Equal, Alloc>>
  : map_caster<std::unordered_map<Key, Value, Hash, Equal, Alloc>, Key, Value> { };

// This type caster is intended to be used for std::optional and std::experimental::optional
template<typename T> struct optional_caster {
    using value_type = typename intrinsic_type<typename T::value_type>::type;
    using caster_type = type_caster<value_type>;

    static handle cast(const T& src, return_value_policy policy, handle parent) {
        if (!src)
            return none().inc_ref();
        return caster_type::cast(*src, policy, parent);
    }

    bool load(handle src, bool convert) {
        if (!src) {
            return false;
        } else if (src.is_none()) {
            value = {};  // nullopt
            return true;
        } else if (!inner.load(src, convert)) {
            return false;
        } else {
            value.emplace(static_cast<const value_type&>(inner));
            return true;
        }
    }

    PYBIND11_TYPE_CASTER(T, _("Optional[") + caster_type::name() + _("]"));

private:
    caster_type inner;
};

#if PYBIND11_HAS_OPTIONAL
template<typename T> struct type_caster<std::optional<T>>
    : public optional_caster<std::optional<T>> {};

template<> struct type_caster<std::nullopt_t>
    : public void_caster<std::nullopt_t> {};
#endif

#if PYBIND11_HAS_EXP_OPTIONAL
template<typename T> struct type_caster<std::experimental::optional<T>>
    : public optional_caster<std::experimental::optional<T>> {};

template<> struct type_caster<std::experimental::nullopt_t>
    : public void_caster<std::experimental::nullopt_t> {};
#endif

NAMESPACE_END(detail)

inline std::ostream &operator<<(std::ostream &os, const handle &obj) {
    os << (std::string) str(obj);
    return os;
}

NAMESPACE_END(pybind11)

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
