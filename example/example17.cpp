/*
    example/example17.cpp -- Usade of stl_binders functions

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"

#include <pybind11/stl_binders.h>

class A
{
public:
	A() = delete;
};

void init_ex17(py::module &m)
{
	pybind11::class_<A>(m, "A");

    py::vector_binder<int>(m, "VectorInt");

    py::vector_binder<A>(m, "VectorA");
}
