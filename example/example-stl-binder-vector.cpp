/*
    example/example-stl-binder-vector.cpp -- Usage of stl_binders functions

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

void init_ex_stl_binder_vector(py::module &m) {
	py::class_<El>(m, "El")
		.def(py::init<int>());

	py::bind_vector<unsigned int>(m, "VectorInt");
	py::bind_vector<bool>(m, "VectorBool");

	py::bind_vector<El>(m, "VectorEl");

    py::bind_vector<std::vector<El>>(m, "VectorVectorEl");
}
