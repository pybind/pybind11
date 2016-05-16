/*
    example/example17.cpp -- Usage of stl_binders functions

    Copyright (c) 2016 Sergey Lyskov

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"

#include <pybind11/stl_bind.h>

class El {
public:
	El() = delete;
	El(int v) : a(v) { }

	int a;
};

std::ostream & operator<<(std::ostream &s, El const&v) {
	s << "El{" << v.a << '}';
	return s;
}

void init_ex17(py::module &m) {
	pybind11::class_<El>(m, "El")
		.def(pybind11::init<int>());

	pybind11::bind_vector<unsigned int>(m, "VectorInt");

	pybind11::bind_vector<El>(m, "VectorEl");

    pybind11::bind_vector<std::vector<El>>(m, "VectorVectorEl");
}
