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


NAMESPACE_BEGIN(pybind11)

#if (!defined _MSC_VER) || (_MSC_VER > 1900)
	#define INSERTION_OPERATOR_IMPLEMENTATION
#endif


template<typename T>
constexpr auto has_equal_operator(int) -> decltype(std::declval<T>() == std::declval<T>(), bool()) { return true; }
template<typename T>
constexpr bool has_equal_operator(...) { return false; }



template<typename T>
constexpr auto has_not_equal_operator(int) -> decltype(std::declval<T>() != std::declval<T>(), bool()) { return true; }
template<typename T>
constexpr bool has_not_equal_operator(...) { return false; }


#ifdef INSERTION_OPERATOR_IMPLEMENTATION
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
#endif


template <typename T, typename Allocator = std::allocator<T>, typename holder_type = std::unique_ptr< std::vector<T, Allocator> > >
class vector_binder {
	using Vector = std::vector<T, Allocator>;
	using SizeType = typename Vector::size_type;

	using Class_ = pybind11::class_<Vector, holder_type >;

	// template<typename U = T, typename std::enable_if< std::is_constructible<U>{} >::type * = nullptr>
	// void maybe_constructible(Class_ &cl) {
	// 	cl.def(pybind11::init<>());
	// }
	// template<typename U = T, typename std::enable_if< !std::is_constructible<U>{} >::type * = nullptr>
	// void maybe_constructible(Class_ &cl) {}

	template<typename U = T, typename std::enable_if< std::is_default_constructible<U>::value >::type * = nullptr>
	void maybe_default_constructible() {
		cl.def(pybind11::init<SizeType>());
		cl.def("resize", (void (Vector::*)(SizeType count)) &Vector::resize, "changes the number of elements stored");

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
	template<typename U = T, typename std::enable_if< !std::is_default_constructible<U>::value >::type * = nullptr>
	void maybe_default_constructible() {}


	template<typename U = T, typename std::enable_if< std::is_copy_constructible<U>::value >::type * = nullptr>
	void maybe_copy_constructible() {
		cl.def(pybind11::init< Vector const &>());
	}
	template<typename U = T, typename std::enable_if< !std::is_copy_constructible<U>::value >::type * = nullptr>
	void maybe_copy_constructible() {}


	template<typename U = T, typename std::enable_if< has_equal_operator<U>(0) >::type * = nullptr>
	void maybe_has_equal_operator() {
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
	template<typename U = T, typename std::enable_if< !has_equal_operator<U>(0) >::type * = nullptr>
	void maybe_has_equal_operator() {}


	template<typename U = T, typename std::enable_if< has_not_equal_operator<U>(0) >::type * = nullptr>
	void maybe_has_not_equal_operator() {
	    cl.def(pybind11::self != pybind11::self);
	}
	template<typename U = T, typename std::enable_if< !has_not_equal_operator<U>(0) >::type * = nullptr>
	void maybe_has_not_equal_operator() {}


	#ifdef INSERTION_OPERATOR_IMPLEMENTATION
	template<typename U = T, typename std::enable_if< has_insertion_operator<U>() >::type * = nullptr>
	void maybe_has_insertion_operator(char const *name) {
		cl.def("__repr__", [name](typename vector_binder<T>::Vector &v) {
				std::ostringstream s;
				s << name << '[';
				for(uint i=0; i<v.size(); ++i) {
					s << v[i];
					if(i != v.size()-1) s << ", ";
				}
				s << ']';
				return s.str();
			});

	}
	template<typename U = T, typename std::enable_if< !has_insertion_operator<U>() >::type * = nullptr>
	void maybe_has_insertion_operator(char const *) {}
	#endif



public:
	Class_ cl;

	vector_binder(pybind11::module &m, char const *name, char const *doc=nullptr) : cl(m, name, doc) {
		cl.def(pybind11::init<>());

		//maybe_constructible(cl);
		maybe_default_constructible();
		maybe_copy_constructible();

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
		maybe_has_equal_operator();
		maybe_has_not_equal_operator();

		// Printing
		#ifdef INSERTION_OPERATOR_IMPLEMENTATION
		maybe_has_insertion_operator(name);
		#endif

		// C++ style functions deprecated, leaving it here as an example
		//cl.def("empty",         &Vector::empty,         "checks whether the container is empty");
		//cl.def("size",          &Vector::size,          "returns the number of elements");
		//cl.def("push_back", (void (Vector::*)(const T&)) &Vector::push_back, "adds an element to the end");
		//cl.def("pop_back",                                &Vector::pop_back, "removes the last element");
	}
};


NAMESPACE_END(pybind11)

#endif // _INCLUDED_std_binders_h_
