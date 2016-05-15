/*
    pybind11/std_binders.h: Convenience wrapper functions for STL containers with C++ like interface

    Copyright (c) 2016 Sergey Lyskov

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#ifndef _INCLUDED_std_binders_h_
#define _INCLUDED_std_binders_h_

#include "common.h"
#include "operators.h"

#include <type_traits>
#include <utility>
#include <algorithm>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <deque>

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

template<typename T>
constexpr auto has_equal_operator(int) -> decltype(std::declval<T>() == std::declval<T>(), std::declval<T>() != std::declval<T>(), bool()) { return true; }
template<typename T>
constexpr bool has_equal_operator(...) { return false; }

template<typename T, typename SFINAE = void>
struct has_equal_operator_s {
	static const bool value = false;
};
template<typename T>
struct has_equal_operator_s<T>
{
	static const bool value = has_equal_operator<T>(0);
};
template<> template <typename A>
struct has_equal_operator_s< std::vector<A> >
{
	static const bool value = has_equal_operator_s<A>::value;
};
template<> template <typename A>
struct has_equal_operator_s< std::deque<A> >
{
	static const bool value = has_equal_operator_s<A>::value;
};
template<> template <typename A>
struct has_equal_operator_s< std::set<A> >
{
	static const bool value = has_equal_operator_s<A>::value;
};
template<> template <typename A, typename B>
struct has_equal_operator_s< std::pair<A,B> >
{
	static const bool value = has_equal_operator_s<A>::value and has_equal_operator_s<B>::value;
};
template<> template <typename A, typename B>
struct has_equal_operator_s< std::map<A,B> >
{
	static const bool value = has_equal_operator_s<A>::value and has_equal_operator_s<B>::value;
};


namespace has_insertion_operator_implementation {
enum class False {};
struct any_type {
    template<typename T> any_type(T const&);
};
False operator<<(std::ostream const&, any_type const&);
}
template<typename T>
constexpr bool has_insertion_operator() {
	using namespace has_insertion_operator_implementation;
	return std::is_same< decltype(std::declval<std::ostream&>() << std::declval<T>()), std::ostream & >::value;
}

// Workaround for MSVC 2015
template<typename T>
struct has_insertion_operator_s {
	static const bool value = has_insertion_operator<T>();
};


template<typename Vector, typename Class_, typename std::enable_if< std::is_default_constructible<typename Vector::value_type>::value >::type * = nullptr>
void vector_maybe_default_constructible(Class_ &cl) {
	using size_type = typename Vector::size_type;

	cl.def(pybind11::init<size_type>());
	cl.def("resize", (void (Vector::*)(size_type count)) &Vector::resize, "changes the number of elements stored");

	/// Slicing protocol
	cl.def("__getitem__", [](Vector const &v, pybind11::slice slice) -> Vector * {
			pybind11::ssize_t start, stop, step, slicelength;
			if(!slice.compute(v.size(), &start, &stop, &step, &slicelength))
				throw pybind11::error_already_set();
			Vector *seq = new Vector(slicelength);
			for (int i=0; i<slicelength; ++i) {
				(*seq)[i] = v[start]; start += step;
			}
			return seq;
		});
}
template<typename Vector, typename Class_, typename std::enable_if< !std::is_default_constructible<typename Vector::value_type>::value >::type * = nullptr>
void vector_maybe_default_constructible(Class_ &) {}


template<typename Vector, typename Class_, typename std::enable_if< std::is_copy_constructible<typename Vector::value_type>::value >::type * = nullptr>
void vector_maybe_copy_constructible(Class_ &cl) {
	cl.def(pybind11::init< Vector const &>());
}
template<typename Vector, typename Class_, typename std::enable_if< !std::is_copy_constructible<typename Vector::value_type>::value >::type * = nullptr>
void vector_maybe_copy_constructible(Class_ &) {}


template<typename Vector, typename Class_, typename std::enable_if< has_equal_operator_s<typename Vector::value_type>::value >::type * = nullptr>
void vector_maybe_has_equal_operator(Class_ &cl) {
	using T = typename Vector::value_type;

	cl.def(pybind11::self == pybind11::self);
	cl.def(pybind11::self != pybind11::self);

	cl.def("count", [](Vector const &v, T const & value) { return std::count(v.begin(), v.end(), value); }, "counts the elements that are equal to value");

	cl.def("remove", [](Vector &v, T const &t) {
			auto p = std::find(v.begin(), v.end(), t);
			if(p != v.end()) v.erase(p);
			else throw pybind11::value_error();
		}, "Remove the first item from the list whose value is x. It is an error if there is no such item.");

	cl.def("__contains__", [](Vector const &v, T const &t) { return std::find(v.begin(), v.end(), t) != v.end(); }, "return true if item in the container");
}
template<typename Vector, typename Class_, typename std::enable_if< !has_equal_operator_s<typename Vector::value_type>::value >::type * = nullptr>
void vector_maybe_has_equal_operator(Class_ &) {}


template<typename Vector, typename Class_, typename std::enable_if< has_insertion_operator_s<typename Vector::value_type>::value >::type * = nullptr>
void vector_maybe_has_insertion_operator(char const *name, Class_ &cl) {
	using size_type = typename Vector::size_type;

	cl.def("__repr__", [name](Vector &v) {
			std::ostringstream s;
			s << name << '[';
			for(size_type i=0; i<v.size(); ++i) {
				s << v[i];
				if(i != v.size()-1) s << ", ";
			}
			s << ']';
			return s.str();
		});
}
template<typename Vector, typename Class_, typename std::enable_if< !has_insertion_operator_s<typename Vector::value_type>::value >::type * = nullptr>
void vector_maybe_has_insertion_operator(char const *, Class_ &) {}


NAMESPACE_END(detail)


template <typename T, typename Allocator = std::allocator<T>, typename holder_type = std::unique_ptr< std::vector<T, Allocator> > >
pybind11::class_<std::vector<T, Allocator>, holder_type > vector_binder(pybind11::module &m, char const *name, char const *doc=nullptr) {
	using Vector = std::vector<T, Allocator>;
	using SizeType = typename Vector::size_type;
	using Class_ = pybind11::class_<Vector, holder_type >;

	Class_ cl(m, name, doc);

	cl.def(pybind11::init<>());

	detail::vector_maybe_default_constructible<Vector>(cl);
	detail::vector_maybe_copy_constructible<Vector>(cl);

	// Element access
	cl.def("front", [](Vector &v) {
			if(v.size()) return v.front();
			else throw pybind11::index_error();
		}, "access the first element");
	cl.def("back", [](Vector &v) {
			if(v.size()) return v.back();
			else throw pybind11::index_error();
		}, "access the last element ");
	// Not needed, the operator[] is already providing bounds checking cl.def("at", (T& (Vector::*)(SizeType i)) &Vector::at, "access specified element with bounds checking");

	// Capacity, C++ style
	cl.def("max_size",      &Vector::max_size,      "returns the maximum possible number of elements");
	cl.def("reserve",       &Vector::reserve,       "reserves storage");
	cl.def("capacity",      &Vector::capacity,      "returns the number of elements that can be held in currently allocated storage");
	cl.def("shrink_to_fit", &Vector::shrink_to_fit, "reduces memory usage by freeing unused memory");

	// Modifiers, C++ style
	cl.def("clear", &Vector::clear, "clears the contents");
	cl.def("swap",   &Vector::swap, "swaps the contents");

	// Modifiers, Python style
	cl.def("append", (void (Vector::*)(const T&)) &Vector::push_back, "adds an element to the end");
	cl.def("insert", [](Vector &v, SizeType i, const T&t) {v.insert(v.begin()+i, t);}, "insert an item at a given position");
	cl.def("extend", [](Vector &v, Vector &src) { v.reserve( v.size() + src.size() ); v.insert(v.end(), src.begin(), src.end()); }, "extend the list by appending all the items in the given vector");
	cl.def("pop", [](Vector &v) {
			if(v.size()) {
				T t = v.back();
				v.pop_back();
				return t;
			}
			else throw pybind11::index_error();
		}, "remove and return last item");

	cl.def("pop", [](Vector &v, SizeType i) {
			if(i >= v.size()) throw pybind11::index_error();
			T t = v[i];
			v.erase(v.begin() + i);
			return t;
		}, "remove and return item at index");

	cl.def("erase", [](Vector &v, SizeType i) {
			if(i >= v.size()) throw pybind11::index_error();
			v.erase(v.begin() + i);
		}, "erases element at index");


	// Python friendly bindings
	#ifdef PYTHON_ABI_VERSION // Python 3+
		cl.def("__bool__",    [](Vector &v) -> bool { return v.size() != 0; }); // checks whether the container has any elements in it
	#else
		cl.def("__nonzero__", [](Vector &v) -> bool { return v.size() != 0; }); // checks whether the container has any elements in it
	#endif

	cl.def("__getitem__", [](Vector const &v, SizeType i) {
			if(i >= v.size()) throw pybind11::index_error();
			return v[i];
		});

	cl.def("__setitem__", [](Vector &v, SizeType i, T const & t) {
			if(i >= v.size()) throw pybind11::index_error();
			v[i] = t;
		});

	cl.def("__len__", &Vector::size);

	cl.def("__iter__", [](Vector &v) { return pybind11::make_iterator(v.begin(), v.end()); },
		   pybind11::keep_alive<0, 1>() /* Essential: keep object alive while iterator exists */);

	/// Slicing protocol
	cl.def("__setitem__", [](Vector &v, pybind11::slice slice,  Vector const &value) {
			pybind11::ssize_t start, stop, step, slicelength;
			if(!slice.compute(v.size(), &start, &stop, &step, &slicelength))
				throw pybind11::error_already_set();
			if((size_t) slicelength != value.size())
				throw std::runtime_error("Left and right hand size of slice assignment have different sizes!");
			for(int i=0; i<slicelength; ++i) {
				v[start] = value[i]; start += step;
			}
		});

	// Comparisons
	detail::vector_maybe_has_equal_operator<Vector>(cl);

	// Printing
	detail::vector_maybe_has_insertion_operator<Vector>(name, cl);

	// C++ style functions deprecated, leaving it here as an example
	//cl.def("empty",         &Vector::empty,         "checks whether the container is empty");
	//cl.def("size",          &Vector::size,          "returns the number of elements");
	//cl.def("push_back", (void (Vector::*)(const T&)) &Vector::push_back, "adds an element to the end");
	//cl.def("pop_back",                                &Vector::pop_back, "removes the last element");

	return cl;
}


NAMESPACE_END(pybind11)

#endif // _INCLUDED_std_binders_h_
