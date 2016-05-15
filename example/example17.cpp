/*
    example/example17.cpp -- Usage of stl_binders functions

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"

#include <pybind11/stl_binders.h>


class A {
public:
	A() = delete;
	A(int v) :a(v) {}

	int a;
};


std::ostream & operator<<(std::ostream &s, A const&v) {
	s << "A{" << v.a << '}';
	return s;
}


void init_ex17(py::module &m) {
	pybind11::class_<A>(m, "A")
		.def(pybind11::init<int>());

	pybind11::vector_binder<int>(m, "VectorInt");

	pybind11::vector_binder<A>(m, "VectorA");

	pybind11::vector_binder< std::vector<A> >(m, "VectorVectorA");
}
