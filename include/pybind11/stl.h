/*
    pybind11/stl.h: Transparent conversion for STL data types

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "pybind11.h"
#include "detail/common.h"
#include "detail/descr.h"
#include "detail/type_caster_base.h"

#include <deque>
#include <initializer_list>
#include <list>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <valarray>

// See `detail/common.h` for implementation of these guards.
#if defined(PYBIND11_HAS_OPTIONAL)
#    include <optional>
#elif defined(PYBIND11_HAS_EXP_OPTIONAL)
#    include <experimental/optional>
#endif

#if defined(PYBIND11_HAS_VARIANT)
#    include <variant>
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

//
// Begin: Equivalent of
//        https://github.com/google/clif/blob/ae4eee1de07cdf115c0c9bf9fec9ff28efce6f6c/clif/python/runtime.cc#L388-L438
/*
The three `object_is_convertible_to_*()` functions below are
the result of converging the behaviors of pybind11 and PyCLIF
(http://github.com/google/clif).

Originally PyCLIF was extremely far on the permissive side of the spectrum,
while pybind11 was very far on the strict side. Originally PyCLIF accepted any
Python iterable as input for a C++ `vector`/`set`/`map` argument, as long as
the elements were convertible. The obvious (in hindsight) problem was that
any empty Python iterable could be passed to any of these C++ types, e.g. `{}`
was accepted for C++ `vector`/`set` arguments, or `[]` for C++ `map` arguments.

The functions below strike a practical permissive-vs-strict compromise,
informed by tens of thousands of use cases in the wild. A main objective is
to prevent accidents and improve readability:

- Python literals must match the C++ types.

- For C++ `set`: The potentially reducing conversion from a Python sequence
  (e.g. Python `list` or `tuple`) to a C++ `set` must be explicit, by going
  through a Python `set`.

- However, a Python `set` can still be passed to a C++ `vector`. The rationale
  is that this conversion is not reducing. Implicit conversions of this kind
  are also fairly commonly used, therefore enforcing explicit conversions
  would have an unfavorable cost : benefit ratio; more sloppily speaking,
  such an enforcement would be more annoying than helpful.

Additional checks have been added to allow types derived from `collections.abc.Set` and
`collections.abc.Mapping` (`collections.abc.Sequence` is already allowed by `PySequence_Check`).
*/

inline bool object_is_instance_with_one_of_tp_names(PyObject *obj,
                                                    std::initializer_list<const char *> tp_names) {
    if (PyType_Check(obj)) {
        return false;
    }
    // A user-defined class can have the same name without being iterable.
    return tp_name_is_one_of(Py_TYPE(obj), tp_names) && PyIterable_Check(obj);
}

inline bool object_is_generator(PyObject *obj) {
#if defined(Py_LIMITED_API)
    // Exact type check, like PyGen_Check(): a user class named `generator` must not match.
    // The import can release the GIL, so a plain function-local static could deadlock.
    PYBIND11_CONSTINIT static gil_safe_call_once_and_store<PyTypeObject *> storage;
    return Py_TYPE(obj)
           == storage
                  .call_once_and_store_result([] {
                      return reinterpret_cast<PyTypeObject *>(
                          object(module_::import("types").attr("GeneratorType")).release().ptr());
                  })
                  .get_stored();
#else
    return PyGen_Check(obj) != 0;
#endif
}

inline bool object_is_convertible_to_std_vector(const handle &src) {
    // Allow sequence-like objects, but not (byte-)string-like objects.
    if (PySequence_Check(src.ptr()) != 0) {
        return !PyUnicode_Check(src.ptr()) && !PyBytes_Check(src.ptr());
    }
    // Allow generators, set/frozenset and several common iterable types.
    return object_is_generator(src.ptr()) || (PyAnySet_Check(src.ptr()) != 0)
           || object_is_instance_with_one_of_tp_names(
               src.ptr(), {"dict_keys", "dict_values", "dict_items", "map", "zip"});
}

inline bool object_is_convertible_to_std_set(const handle &src, bool convert) {
    // Allow set/frozenset and dict keys.
    // In convert mode: also allow types derived from collections.abc.Set.
    return ((PyAnySet_Check(src.ptr()) != 0)
            || object_is_instance_with_one_of_tp_names(src.ptr(), {"dict_keys"}))
           || (convert && isinstance(src, module_::import("collections.abc").attr("Set")));
}

inline bool object_is_convertible_to_std_map(const handle &src, bool convert) {
    // Allow dict.
    if (PyDict_Check(src.ptr())) {
        return true;
    }
    // Allow types conforming to Mapping Protocol.
    // According to https://docs.python.org/3/c-api/mapping.html, `PyMappingCheck()` checks for
    // `__getitem__()` without checking the type of keys. In order to restrict the allowed types
    // closer to actual Mapping-like types, we also check for the `items()` method.
    if (PyMapping_Check(src.ptr()) != 0) {
        PyObject *items = PyObject_GetAttrString(src.ptr(), "items");
        if (items != nullptr) {
            bool is_convertible = (PyCallable_Check(items) != 0);
            Py_DECREF(items);
            if (is_convertible) {
                return true;
            }
        } else {
            PyErr_Clear();
        }
    }
    // In convert mode: Allow types derived from collections.abc.Mapping
    return convert && isinstance(src, module_::import("collections.abc").attr("Mapping"));
}

//
// End: Equivalent of clif/python/runtime.cc
//

/// Extracts an const lvalue reference or rvalue reference for U based on the type of T (e.g. for
/// forwarding a container element).  Typically used indirect via forwarded_type(), below.
template <typename T, typename U>
using forwarded_type = conditional_t<std::is_lvalue_reference<T>::value,
                                     remove_reference_t<U> &,
                                     remove_reference_t<U> &&>;

/// Forwards a value U as rvalue or lvalue according to whether T is rvalue or lvalue; typically
/// used for forwarding a container's elements.
template <typename T, typename U>
constexpr forwarded_type<T, U> forward_like(U &&u) {
    return std::forward<detail::forwarded_type<T, U>>(std::forward<U>(u));
}

// Checks if a container has a STL style reserve method.
// This will only return true for a `reserve()` with a `void` return.
template <typename C>
using has_reserve_method = std::is_same<decltype(std::declval<C>().reserve(0)), void>;

template <typename Type, typename Key>
struct set_caster {
    using type = Type;
    using key_conv = make_caster<Key>;

private:
    template <typename T = Type, enable_if_t<has_reserve_method<T>::value, int> = 0>
    void reserve_maybe(const anyset &s, Type *) {
        value.reserve(s.size());
    }
    void reserve_maybe(const anyset &, void *) {}

    bool convert_iterable(const iterable &itbl, bool convert) {
        for (const auto &it : itbl) {
            key_conv conv;
            if (!conv.load(it, convert)) {
                return false;
            }
            value.insert(cast_op<Key &&>(std::move(conv)));
        }
        return true;
    }

    bool convert_anyset(const anyset &s, bool convert) {
        value.clear();
        reserve_maybe(s, &value);
        return convert_iterable(s, convert);
    }

public:
    bool load(handle src, bool convert) {
        if (!object_is_convertible_to_std_set(src, convert)) {
            return false;
        }
        if (isinstance<anyset>(src)) {
            value.clear();
            return convert_anyset(reinterpret_borrow<anyset>(src), convert);
        }
        if (!convert) {
            return false;
        }
        assert(isinstance<iterable>(src));
        value.clear();
        return convert_iterable(reinterpret_borrow<iterable>(src), convert);
    }

    template <typename T>
    static handle cast(T &&src, return_value_policy policy, handle parent) {
        if (!std::is_lvalue_reference<T>::value) {
            policy = return_value_policy_override<Key>::policy(policy);
        }
        pybind11::set s;
        for (auto &&value : src) {
            auto value_ = reinterpret_steal<object>(
                key_conv::cast(detail::forward_like<T>(value), policy, parent));
            if (!value_ || !s.add(std::move(value_))) {
                return handle();
            }
        }
        return s.release();
    }

    PYBIND11_TYPE_CASTER(type,
                         io_name("collections.abc.Set", "set") + const_name("[") + key_conv::name
                             + const_name("]"));
};

template <typename Type, typename Key, typename Value>
struct map_caster {
    using key_conv = make_caster<Key>;
    using value_conv = make_caster<Value>;

private:
    template <typename T = Type, enable_if_t<has_reserve_method<T>::value, int> = 0>
    void reserve_maybe(const dict &d, Type *) {
        value.reserve(d.size());
    }
    void reserve_maybe(const dict &, void *) {}

    bool convert_elements(const dict &d, bool convert) {
        value.clear();
        reserve_maybe(d, &value);
        for (const auto &it : d) {
            key_conv kconv;
            value_conv vconv;
            if (!kconv.load(it.first.ptr(), convert) || !vconv.load(it.second.ptr(), convert)) {
                return false;
            }
            value.emplace(cast_op<Key &&>(std::move(kconv)), cast_op<Value &&>(std::move(vconv)));
        }
        return true;
    }

public:
    bool load(handle src, bool convert) {
        if (!object_is_convertible_to_std_map(src, convert)) {
            return false;
        }
        if (isinstance<dict>(src)) {
            return convert_elements(reinterpret_borrow<dict>(src), convert);
        }
        if (!convert) {
            return false;
        }
        auto items = reinterpret_steal<object>(PyMapping_Items(src.ptr()));
        if (!items) {
            throw error_already_set();
        }
        assert(isinstance<iterable>(items));
        return convert_elements(dict(reinterpret_borrow<iterable>(items)), convert);
    }

    template <typename T>
    static handle cast(T &&src, return_value_policy policy, handle parent) {
        dict d;
        return_value_policy policy_key = policy;
        return_value_policy policy_value = policy;
        if (!std::is_lvalue_reference<T>::value) {
            policy_key = return_value_policy_override<Key>::policy(policy_key);
            policy_value = return_value_policy_override<Value>::policy(policy_value);
        }
        for (auto &&kv : src) {
            auto key = reinterpret_steal<object>(
                key_conv::cast(detail::forward_like<T>(kv.first), policy_key, parent));
            auto value = reinterpret_steal<object>(
                value_conv::cast(detail::forward_like<T>(kv.second), policy_value, parent));
            if (!key || !value) {
                return handle();
            }
            d[std::move(key)] = std::move(value);
        }
        return d.release();
    }

    PYBIND11_TYPE_CASTER(Type,
                         io_name("collections.abc.Mapping", "dict") + const_name("[")
                             + key_conv::name + const_name(", ") + value_conv::name
                             + const_name("]"));
};

template <typename Type, typename Value>
struct list_caster {
    using value_conv = make_caster<Value>;

    bool load(handle src, bool convert) {
        if (!object_is_convertible_to_std_vector(src)) {
            return false;
        }
        if (isinstance<sequence>(src)) {
            return convert_elements(src, convert);
        }
        if (!convert) {
            return false;
        }
        // Designed to be behavior-equivalent to passing tuple(src) from Python:
        // The conversion to a tuple will first exhaust the generator object, to ensure that
        // the generator is not left in an unpredictable (to the caller) partially-consumed
        // state.
        assert(isinstance<iterable>(src));
        return convert_elements(tuple(reinterpret_borrow<iterable>(src)), convert);
    }

private:
    template <typename T = Type, enable_if_t<has_reserve_method<T>::value, int> = 0>
    void reserve_maybe(const sequence &s, Type *) {
        value.reserve(s.size());
    }
    void reserve_maybe(const sequence &, void *) {}

    bool convert_elements(handle seq, bool convert) {
        auto s = reinterpret_borrow<sequence>(seq);
        value.clear();
        reserve_maybe(s, &value);
        for (const auto &it : seq) {
            value_conv conv;
            if (!conv.load(it, convert)) {
                return false;
            }
            value.push_back(cast_op<Value &&>(std::move(conv)));
        }
        return true;
    }

public:
    template <typename T>
    static handle cast(T &&src, return_value_policy policy, handle parent) {
        if (!std::is_lvalue_reference<T>::value) {
            policy = return_value_policy_override<Value>::policy(policy);
        }
        list l(src.size());
        ssize_t index = 0;
        for (auto &&value : src) {
            auto value_ = reinterpret_steal<object>(
                value_conv::cast(detail::forward_like<T>(value), policy, parent));
            if (!value_) {
                return handle();
            }
            list_set_item(l.ptr(), index++, value_.release().ptr()); // steals a reference
        }
        return l.release();
    }

    PYBIND11_TYPE_CASTER(Type,
                         io_name("collections.abc.Sequence", "list") + const_name("[")
                             + value_conv::name + const_name("]"));
};

template <typename Type, typename Alloc>
struct type_caster<std::vector<Type, Alloc>> : list_caster<std::vector<Type, Alloc>, Type> {};

template <typename Type, typename Alloc>
struct type_caster<std::deque<Type, Alloc>> : list_caster<std::deque<Type, Alloc>, Type> {};

template <typename Type, typename Alloc>
struct type_caster<std::list<Type, Alloc>> : list_caster<std::list<Type, Alloc>, Type> {};

template <typename ArrayType, typename V, size_t... I>
ArrayType vector_to_array_impl(V &&v, index_sequence<I...>) {
    return {{std::move(v[I])...}};
}

// Based on https://en.cppreference.com/w/cpp/container/array/to_array
template <typename ArrayType, size_t N, typename V>
ArrayType vector_to_array(V &&v) {
    return vector_to_array_impl<ArrayType, V>(std::forward<V>(v), make_index_sequence<N>{});
}

template <typename ArrayType, typename Value, bool Resizable, size_t Size = 0>
struct array_caster {
    using value_conv = make_caster<Value>;

private:
    // On the heap because `ArrayType` is not necessarily default constructible.
    std::unique_ptr<ArrayType> value;

    // A `std::array` can only be populated in one of two ways:
    // * Aggregate-initialize it from `Size` initializers, which copy or move constructs the
    //   elements. This needs a temporary `std::vector` and an `index_sequence` of `Size`
    //   elements, which is very slow to compile for large `Size` (#5765).
    // * Default-construct it, then assign the elements, as for `std::valarray`. In general that
    //   is not equivalent: a default constructor can have side effects, and an assignment
    //   operator does not necessarily do what the copy constructor does (`boost::multi_array`,
    //   for example, does not resize on assignment). It is therefore used only for `Value`
    //   types for which it cannot be told apart from constructing the elements.
    bool convert_elements(handle seq, bool convert) {
        // What `cast_op<Value &&>` yields, e.g. `Value &` for types registered with `py::class_`.
        // Not in class scope: a `value_conv` that is only used for `cast()` does not need to
        // have a `cast_op_type`.
        using value_cast_op_type = typename value_conv::template cast_op_type<Value &&>;
        // The conditions are on `Value` rather than `ArrayType`, to keep the selection
        // independent of `Size`: a `std::array` with `Size == 0` can be trivially default
        // constructible even if `Value` is not.
        using assign_elements = bool_constant<
            Resizable
            || (std::is_trivially_default_constructible<Value>::value
                && std::is_trivially_copyable<Value>::value
                && std::is_trivially_assignable<Value &, value_cast_op_type>::value)>;
        return convert_elements(seq, convert, assign_elements{});
    }

    // Returns a default-constructed array for `size` elements, or `nullptr` if `size` is not
    // acceptable. `ArrayType()` rather than `ArrayType{}`: the latter is ill-formed for `Value`
    // types with an explicit default constructor.
    static std::unique_ptr<ArrayType> make_array(size_t size, std::true_type /*Resizable*/) {
        std::unique_ptr<ArrayType> array(new ArrayType());
        // For the `resize` to work, `Value` must be default constructible.
        // For `std::valarray`, this is a requirement:
        // https://en.cppreference.com/w/cpp/named_req/NumericType
        array->resize(size);
        return array;
    }
    static std::unique_ptr<ArrayType> make_array(size_t size, std::false_type /*Resizable*/) {
        return std::unique_ptr<ArrayType>(size == Size ? new ArrayType() : nullptr);
    }

    // Both overloads below index `l` up to a size that is determined only once, rather than
    // iterating over `l`: the accesses stay in bounds even if the length of `l` changes
    // underneath (e.g. if another thread resizes a list in a free-threaded build).

    bool convert_elements(handle seq, bool convert, std::true_type /*assign_elements*/) {
        auto l = reinterpret_borrow<sequence>(seq);
        const size_t size = l.size();
        auto result = make_array(size, bool_constant<Resizable>{});
        if (!result) {
            return false;
        }
        for (size_t i = 0; i < size; i++) {
            const object item = l[i]; // Keeps the item alive while `conv` is in use.
            value_conv conv;
            if (!conv.load(item, convert)) {
                return false;
            }
            (*result)[i] = cast_op<Value &&>(std::move(conv));
        }
        value = std::move(result);
        return true;
    }

    bool convert_elements(handle seq, bool convert, std::false_type /*assign_elements*/) {
        auto l = reinterpret_borrow<sequence>(seq);
        const size_t size = l.size();
        if (size != Size) {
            return false;
        }
        std::vector<Value> temp;
        temp.reserve(size);
        for (size_t i = 0; i < size; i++) {
            const object item = l[i]; // Keeps the item alive while `conv` is in use.
            value_conv conv;
            if (!conv.load(item, convert)) {
                return false;
            }
            temp.emplace_back(cast_op<Value &&>(std::move(conv)));
        }
        value.reset(new ArrayType(vector_to_array<ArrayType, Size>(std::move(temp))));
        return true;
    }

public:
    bool load(handle src, bool convert) {
        if (!object_is_convertible_to_std_vector(src)) {
            return false;
        }
        if (isinstance<sequence>(src)) {
            return convert_elements(src, convert);
        }
        if (!convert) {
            return false;
        }
        // Designed to be behavior-equivalent to passing tuple(src) from Python:
        // The conversion to a tuple will first exhaust the generator object, to ensure that
        // the generator is not left in an unpredictable (to the caller) partially-consumed
        // state.
        assert(isinstance<iterable>(src));
        return convert_elements(tuple(reinterpret_borrow<iterable>(src)), convert);
    }

    template <typename T>
    static handle cast(T &&src, return_value_policy policy, handle parent) {
        list l(src.size());
        ssize_t index = 0;
        for (auto &&value : src) {
            auto value_ = reinterpret_steal<object>(
                value_conv::cast(detail::forward_like<T>(value), policy, parent));
            if (!value_) {
                return handle();
            }
            list_set_item(l.ptr(), index++, value_.release().ptr()); // steals a reference
        }
        return l.release();
    }

    // Code copied from PYBIND11_TYPE_CASTER macro.
    // Intentionally preserving the behavior exactly as it was before PR #5305
    template <typename T_, enable_if_t<std::is_same<ArrayType, remove_cv_t<T_>>::value, int> = 0>
    static handle cast(T_ *src, return_value_policy policy, handle parent) {
        if (!src) {
            return none().release();
        }
        if (policy == return_value_policy::take_ownership) {
            auto h = cast(std::move(*src), policy, parent);
            delete src; // WARNING: Assumes `src` was allocated with `new`.
            return h;
        }
        return cast(*src, policy, parent);
    }

    // NOLINTNEXTLINE(google-explicit-constructor)
    operator ArrayType *() { return &(*value); }
    // NOLINTNEXTLINE(google-explicit-constructor)
    operator ArrayType &() { return *value; }
    // NOLINTNEXTLINE(google-explicit-constructor)
    operator ArrayType &&() && { return std::move(*value); }

    template <typename T_>
    using cast_op_type = movable_cast_op_type<T_>;

    static constexpr auto name
        = const_name<Resizable>(const_name(""), const_name("typing.Annotated["))
          + io_name("collections.abc.Sequence", "list") + const_name("[") + value_conv::name
          + const_name("]")
          + const_name<Resizable>(const_name(""),
                                  const_name(", \"FixedSize(") + const_name<Size>()
                                      + const_name(")\"]"));
};

template <typename Type, size_t Size>
struct type_caster<std::array<Type, Size>>
    : array_caster<std::array<Type, Size>, Type, false, Size> {};

template <typename Type>
struct type_caster<std::valarray<Type>> : array_caster<std::valarray<Type>, Type, true> {};

template <typename Key, typename Compare, typename Alloc>
struct type_caster<std::set<Key, Compare, Alloc>>
    : set_caster<std::set<Key, Compare, Alloc>, Key> {};

template <typename Key, typename Hash, typename Equal, typename Alloc>
struct type_caster<std::unordered_set<Key, Hash, Equal, Alloc>>
    : set_caster<std::unordered_set<Key, Hash, Equal, Alloc>, Key> {};

template <typename Key, typename Value, typename Compare, typename Alloc>
struct type_caster<std::map<Key, Value, Compare, Alloc>>
    : map_caster<std::map<Key, Value, Compare, Alloc>, Key, Value> {};

template <typename Key, typename Value, typename Hash, typename Equal, typename Alloc>
struct type_caster<std::unordered_map<Key, Value, Hash, Equal, Alloc>>
    : map_caster<std::unordered_map<Key, Value, Hash, Equal, Alloc>, Key, Value> {};

// This type caster is intended to be used for std::optional and std::experimental::optional
template <typename Type, typename Value = typename Type::value_type>
struct optional_caster {
    using value_conv = make_caster<Value>;

    template <typename T>
    static handle cast(T &&src, return_value_policy policy, handle parent) {
        if (!src) {
            return none().release();
        }
        if (!std::is_lvalue_reference<T>::value) {
            policy = return_value_policy_override<Value>::policy(policy);
        }
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        return value_conv::cast(*std::forward<T>(src), policy, parent);
    }

    bool load(handle src, bool convert) {
        if (!src) {
            return false;
        }
        if (src.is_none()) {
            return true; // default-constructed value is already empty
        }
        value_conv inner_caster;
        if (!inner_caster.load(src, convert)) {
            return false;
        }

        value.emplace(cast_op<Value &&>(std::move(inner_caster)));
        return true;
    }

    PYBIND11_TYPE_CASTER(Type, value_conv::name | make_caster<none>::name);
};

#if defined(PYBIND11_HAS_OPTIONAL)
template <typename T>
struct type_caster<std::optional<T>> : public optional_caster<std::optional<T>> {};

template <>
struct type_caster<std::nullopt_t> : public void_caster<std::nullopt_t> {};
#endif

#if defined(PYBIND11_HAS_EXP_OPTIONAL)
template <typename T>
struct type_caster<std::experimental::optional<T>>
    : public optional_caster<std::experimental::optional<T>> {};

template <>
struct type_caster<std::experimental::nullopt_t>
    : public void_caster<std::experimental::nullopt_t> {};
#endif

/// Visit a variant and cast any found type to Python
struct variant_caster_visitor {
    return_value_policy policy;
    handle parent;

    using result_type = handle; // required by boost::variant in C++11

    template <typename T>
    result_type operator()(T &&src) const {
        return make_caster<T>::cast(std::forward<T>(src), policy, parent);
    }
};

/// Helper class which abstracts away variant's `visit` function. `std::variant` and similar
/// `namespace::variant` types which provide a `namespace::visit()` function are handled here
/// automatically using argument-dependent lookup. Users can provide specializations for other
/// variant-like classes, e.g. `boost::variant` and `boost::apply_visitor`.
template <template <typename...> class Variant>
struct visit_helper {
    template <typename... Args>
    static auto call(Args &&...args) -> decltype(visit(std::forward<Args>(args)...)) {
        return visit(std::forward<Args>(args)...);
    }
};

/// Generic variant caster
template <typename Variant>
struct variant_caster;

template <template <typename...> class V, typename... Ts>
struct variant_caster<V<Ts...>> {
    static_assert(sizeof...(Ts) > 0, "Variant must consist of at least one alternative.");

    template <typename U, typename... Us>
    bool load_alternative(handle src, bool convert, type_list<U, Us...>) {
        auto caster = make_caster<U>();
        if (caster.load(src, convert)) {
            value = cast_op<U>(std::move(caster));
            return true;
        }
        return load_alternative(src, convert, type_list<Us...>{});
    }

    bool load_alternative(handle, bool, type_list<>) { return false; }

    bool load(handle src, bool convert) {
        // Do a first pass without conversions to improve constructor resolution.
        // E.g. `py::int_(1).cast<variant<double, int>>()` needs to fill the `int`
        // slot of the variant. Without two-pass loading `double` would be filled
        // because it appears first and a conversion is possible.
        if (convert && load_alternative(src, false, type_list<Ts...>{})) {
            return true;
        }
        return load_alternative(src, convert, type_list<Ts...>{});
    }

    template <typename Variant>
    static handle cast(Variant &&src, return_value_policy policy, handle parent) {
        return visit_helper<V>::call(variant_caster_visitor{policy, parent},
                                     std::forward<Variant>(src));
    }

    using Type = V<Ts...>;
    PYBIND11_TYPE_CASTER(Type, ::pybind11::detail::union_concat(make_caster<Ts>::name...));
};

#if defined(PYBIND11_HAS_VARIANT)
template <typename... Ts>
struct type_caster<std::variant<Ts...>> : variant_caster<std::variant<Ts...>> {};

template <>
struct type_caster<std::monostate> : public void_caster<std::monostate> {};
#endif

PYBIND11_NAMESPACE_END(detail)

inline std::ostream &operator<<(std::ostream &os, const handle &obj) {
#ifdef PYBIND11_HAS_STRING_VIEW
    os << str(obj).cast<std::string_view>();
#else
    os << (std::string) str(obj);
#endif
    return os;
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
