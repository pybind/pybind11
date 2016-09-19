/*
    pybind11/std_bind.h: Binding generators for STL data types

    Copyright (c) 2016 Sergey Lyskov and Wenzel Jakob

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"
#include "operators.h"

#include <algorithm>
#include <sstream>

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

/* SFINAE helper class used by 'is_comparable */
template <typename T>  struct container_traits {
    template <typename T2> static std::true_type test_comparable(decltype(std::declval<const T2 &>() == std::declval<const T2 &>())*);
    template <typename T2> static std::false_type test_comparable(...);
    template <typename T2> static std::true_type test_value(typename T2::value_type *);
    template <typename T2> static std::false_type test_value(...);
    template <typename T2> static std::true_type test_pair(typename T2::first_type *, typename T2::second_type *);
    template <typename T2> static std::false_type test_pair(...);

    static constexpr const bool is_comparable = std::is_same<std::true_type, decltype(test_comparable<T>(nullptr))>::value;
    static constexpr const bool is_pair = std::is_same<std::true_type, decltype(test_pair<T>(nullptr, nullptr))>::value;
    static constexpr const bool is_vector = std::is_same<std::true_type, decltype(test_value<T>(nullptr))>::value;
    static constexpr const bool is_element = !is_pair && !is_vector;
};

/* Default: is_comparable -> std::false_type */
template <typename T, typename SFINAE = void>
struct is_comparable : std::false_type { };

/* For non-map data structures, check whether operator== can be instantiated */
template <typename T>
struct is_comparable<
    T, enable_if_t<container_traits<T>::is_element &&
                   container_traits<T>::is_comparable>>
    : std::true_type { };

/* For a vector/map data structure, recursively check the value type (which is std::pair for maps) */
template <typename T>
struct is_comparable<T, enable_if_t<container_traits<T>::is_vector>> {
    static constexpr const bool value =
        is_comparable<typename T::value_type>::value;
};

/* For pairs, recursively check the two data types */
template <typename T>
struct is_comparable<T, enable_if_t<container_traits<T>::is_pair>> {
    static constexpr const bool value =
        is_comparable<typename T::first_type>::value &&
        is_comparable<typename T::second_type>::value;
};

/* Fallback functions */
template <typename, typename, typename... Args> void vector_if_copy_constructible(const Args&...) { }
template <typename, typename, typename... Args> void vector_if_equal_operator(const Args&...) { }
template <typename, typename, typename... Args> void vector_if_insertion_operator(const Args&...) { }

template<typename Vector, typename Class_, enable_if_t<std::is_copy_constructible<typename Vector::value_type>::value, int> = 0>
void vector_if_copy_constructible(Class_ &cl) {
    cl.def(pybind11::init<const Vector &>(),
           "Copy constructor");
}

template<typename Vector, typename Class_, enable_if_t<is_comparable<Vector>::value, int> = 0>
void vector_if_equal_operator(Class_ &cl) {
    using T = typename Vector::value_type;

    cl.def(self == self);
    cl.def(self != self);

    cl.def("count",
        [](const Vector &v, const T &x) {
            return std::count(v.begin(), v.end(), x);
        },
        arg("x"),
        "Return the number of times ``x`` appears in the list"
    );

    cl.def("remove", [](Vector &v, const T &x) {
            auto p = std::find(v.begin(), v.end(), x);
            if (p != v.end())
                v.erase(p);
            else
                throw pybind11::value_error();
        },
        arg("x"),
        "Remove the first item from the list whose value is x. "
        "It is an error if there is no such item."
    );

    cl.def("__contains__",
        [](const Vector &v, const T &x) {
            return std::find(v.begin(), v.end(), x) != v.end();
        },
        arg("x"),
        "Return true the container contains ``x``"
    );
}

template <typename Vector, typename Class_> auto vector_if_insertion_operator(Class_ &cl, std::string const &name)
    -> decltype(std::declval<std::ostream&>() << std::declval<typename Vector::value_type>(), void()) {
    using size_type = typename Vector::size_type;

    cl.def("__repr__",
           [name](Vector &v) {
            std::ostringstream s;
            s << name << '[';
            for (size_type i=0; i < v.size(); ++i) {
                s << v[i];
                if (i != v.size() - 1)
                    s << ", ";
            }
            s << ']';
            return s.str();
        },
        "Return the canonical string representation of this list."
    );
}

NAMESPACE_END(detail)

//
// std::vector
//
template <typename Vector, typename holder_type = std::unique_ptr<Vector>, typename... Args>
pybind11::class_<Vector, holder_type> bind_vector(pybind11::module &m, std::string const &name, Args&&... args) {
    using T = typename Vector::value_type;
    using SizeType = typename Vector::size_type;
    using DiffType = typename Vector::difference_type;
    using ItType   = typename Vector::iterator;
    using Class_ = pybind11::class_<Vector, holder_type>;

    Class_ cl(m, name.c_str(), std::forward<Args>(args)...);

    cl.def(pybind11::init<>());

    // Register copy constructor (if possible)
    detail::vector_if_copy_constructible<Vector, Class_>(cl);

    // Register comparison-related operators and functions (if possible)
    detail::vector_if_equal_operator<Vector, Class_>(cl);

    // Register stream insertion operator (if possible)
    detail::vector_if_insertion_operator<Vector, Class_>(cl, name);

    cl.def("__init__", [](Vector &v, iterable it) {
        new (&v) Vector();
        try {
            v.reserve(len(it));
            for (handle h : it)
               v.push_back(h.cast<typename Vector::value_type>());
        } catch (...) {
            v.~Vector();
            throw;
        }
    });

    cl.def("append",
           [](Vector &v, const T &value) { v.push_back(value); },
           arg("x"),
           "Add an item to the end of the list");

    cl.def("extend",
       [](Vector &v, Vector &src) {
           v.reserve(v.size() + src.size());
           v.insert(v.end(), src.begin(), src.end());
       },
       arg("L"),
       "Extend the list by appending all the items in the given list"
    );

    cl.def("insert",
        [](Vector &v, SizeType i, const T &x) {
            v.insert(v.begin() + (DiffType) i, x);
        },
        arg("i") , arg("x"),
        "Insert an item at a given position."
    );

    cl.def("pop",
        [](Vector &v) {
            if (v.empty())
                throw pybind11::index_error();
            T t = v.back();
            v.pop_back();
            return t;
        },
        "Remove and return the last item"
    );

    cl.def("pop",
        [](Vector &v, SizeType i) {
            if (i >= v.size())
                throw pybind11::index_error();
            T t = v[i];
            v.erase(v.begin() + (DiffType) i);
            return t;
        },
        arg("i"),
        "Remove and return the item at index ``i``"
    );

    cl.def("__bool__",
        [](const Vector &v) -> bool {
            return !v.empty();
        },
        "Check whether the list is nonempty"
    );

    cl.def("__getitem__",
        [](const Vector &v, SizeType i) -> T {
            if (i >= v.size())
                throw pybind11::index_error();
            return v[i];
        }
    );

    cl.def("__setitem__",
        [](Vector &v, SizeType i, const T &t) {
            if (i >= v.size())
                throw pybind11::index_error();
            v[i] = t;
        }
    );

    cl.def("__delitem__",
        [](Vector &v, SizeType i) {
            if (i >= v.size())
                throw pybind11::index_error();
            v.erase(v.begin() + typename Vector::difference_type(i));
        },
        "Delete list elements using a slice object"
    );

    cl.def("__len__", &Vector::size);

    cl.def("__iter__",
           [](Vector &v) {
               return pybind11::make_iterator<
                   return_value_policy::reference_internal, ItType, ItType, T>(
                   v.begin(), v.end());
           },
           pybind11::keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
    );

    /// Slicing protocol
    cl.def("__getitem__",
        [](const Vector &v, slice slice) -> Vector * {
            size_t start, stop, step, slicelength;

            if (!slice.compute(v.size(), &start, &stop, &step, &slicelength))
                throw pybind11::error_already_set();

            Vector *seq = new Vector();
            seq->reserve((size_t) slicelength);

            for (size_t i=0; i<slicelength; ++i) {
                seq->push_back(v[start]);
                start += step;
            }
            return seq;
        },
        arg("s"),
        "Retrieve list elements using a slice object"
    );

    cl.def("__setitem__",
        [](Vector &v, slice slice,  const Vector &value) {
            size_t start, stop, step, slicelength;
            if (!slice.compute(v.size(), &start, &stop, &step, &slicelength))
                throw pybind11::error_already_set();

            if (slicelength != value.size())
                throw std::runtime_error("Left and right hand size of slice assignment have different sizes!");

            for (size_t i=0; i<slicelength; ++i) {
                v[start] = value[i];
                start += step;
            }
        },
        "Assign list elements using a slice object"
    );

    cl.def("__delitem__",
        [](Vector &v, slice slice) {
            size_t start, stop, step, slicelength;

            if (!slice.compute(v.size(), &start, &stop, &step, &slicelength))
                throw pybind11::error_already_set();

            if (step == 1 && false) {
                v.erase(v.begin() + (DiffType) start, v.begin() + DiffType(start + slicelength));
            } else {
                for (size_t i = 0; i < slicelength; ++i) {
                    v.erase(v.begin() + DiffType(start));
                    start += step - 1;
                }
            }
        },
        "Delete list elements using a slice object"
    );

#if 0
    // C++ style functions deprecated, leaving it here as an example
    cl.def(pybind11::init<size_type>());

    cl.def("resize",
         (void (Vector::*) (size_type count)) & Vector::resize,
         "changes the number of elements stored");

    cl.def("erase",
        [](Vector &v, SizeType i) {
        if (i >= v.size())
            throw pybind11::index_error();
        v.erase(v.begin() + i);
    }, "erases element at index ``i``");

    cl.def("empty",         &Vector::empty,         "checks whether the container is empty");
    cl.def("size",          &Vector::size,          "returns the number of elements");
    cl.def("push_back", (void (Vector::*)(const T&)) &Vector::push_back, "adds an element to the end");
    cl.def("pop_back",                               &Vector::pop_back, "removes the last element");

    cl.def("max_size",      &Vector::max_size,      "returns the maximum possible number of elements");
    cl.def("reserve",       &Vector::reserve,       "reserves storage");
    cl.def("capacity",      &Vector::capacity,      "returns the number of elements that can be held in currently allocated storage");
    cl.def("shrink_to_fit", &Vector::shrink_to_fit, "reduces memory usage by freeing unused memory");

    cl.def("clear", &Vector::clear, "clears the contents");
    cl.def("swap",   &Vector::swap, "swaps the contents");

    cl.def("front", [](Vector &v) {
        if (v.size()) return v.front();
        else throw pybind11::index_error();
    }, "access the first element");

    cl.def("back", [](Vector &v) {
        if (v.size()) return v.back();
        else throw pybind11::index_error();
    }, "access the last element ");

#endif

    return cl;
}



//
// std::map, std::unordered_map
//

NAMESPACE_BEGIN(detail)

/* Fallback functions */
template <typename, typename, typename... Args> void map_if_insertion_operator(const Args&...) { }

template <typename Map, typename Class_, typename... Args> void map_if_copy_assignable(Class_ &cl, const Args&...) {
    using KeyType = typename Map::key_type;
    using MappedType = typename Map::mapped_type;

    cl.def("__setitem__",
           [](Map &m, const KeyType &k, const MappedType &v) {
               auto it = m.find(k);
               if (it != m.end()) it->second = v;
               else m.emplace(k, v);
           }
    );
}

template<typename Map, typename Class_, enable_if_t<!std::is_copy_assignable<typename Map::mapped_type>::value, int> = 0>
void map_if_copy_assignable(Class_ &cl) {
    using KeyType = typename Map::key_type;
    using MappedType = typename Map::mapped_type;

    cl.def("__setitem__",
           [](Map &m, const KeyType &k, const MappedType &v) {
               // We can't use m[k] = v; because value type might not be default constructable
               auto r = m.insert(std::make_pair(k, v));
               if (!r.second) {
                   // value type might be const so the only way to insert it is to erase it first...
                   m.erase(r.first);
                   m.insert(std::make_pair(k, v));
               }
           }
    );
}


template <typename Map, typename Class_> auto map_if_insertion_operator(Class_ &cl, std::string const &name)
-> decltype(std::declval<std::ostream&>() << std::declval<typename Map::key_type>() << std::declval<typename Map::mapped_type>(), void()) {

    cl.def("__repr__",
           [name](Map &m) {
            std::ostringstream s;
            s << name << '{';
            bool f = false;
            for (auto const &kv : m) {
                if (f)
                    s << ", ";
                s << kv.first << ": " << kv.second;
                f = true;
            }
            s << '}';
            return s.str();
        },
        "Return the canonical string representation of this map."
    );
}
NAMESPACE_END(detail)

template <typename Map, typename holder_type = std::unique_ptr<Map>, typename... Args>
pybind11::class_<Map, holder_type> bind_map(module &m, const std::string &name, Args&&... args) {
    using KeyType = typename Map::key_type;
    using MappedType = typename Map::mapped_type;
    using Class_ = pybind11::class_<Map, holder_type>;

    Class_ cl(m, name.c_str(), std::forward<Args>(args)...);

    cl.def(pybind11::init<>());

    // Register stream insertion operator (if possible)
    detail::map_if_insertion_operator<Map, Class_>(cl, name);

    cl.def("__bool__",
        [](const Map &m) -> bool { return !m.empty(); },
        "Check whether the map is nonempty"
    );

    cl.def("__iter__",
           [](Map &m) { return pybind11::make_key_iterator(m.begin(), m.end()); },
           pybind11::keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
    );

    cl.def("items",
           [](Map &m) { return pybind11::make_iterator(m.begin(), m.end()); },
           pybind11::keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
    );

    cl.def("__getitem__",
           [](Map &m, const KeyType &k) -> MappedType {
               auto it = m.find(k);
               if (it == m.end())
                  throw pybind11::key_error();
               return it->second;
           }
    );

    detail::map_if_copy_assignable<Map, Class_>(cl);

    cl.def("__delitem__",
           [](Map &m, const KeyType &k) {
               auto it = m.find(k);
               if (it == m.end())
                   throw pybind11::key_error();
               return m.erase(it);
           }
    );

    cl.def("__len__", &Map::size);

    return cl;
}

NAMESPACE_END(pybind11)
