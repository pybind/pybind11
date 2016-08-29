/*
    tests/test_numpy_array.cpp -- test core array functionality

    Copyright (c) 2016 Ivan Smirnov <i.s.smirnov@gmail.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

test_initializer numpy_array([](py::module &m) {
    m.def("get_arr_ndim", [](const py::array& arr) {
        return arr.ndim();
    });
    m.def("get_arr_shape", [](const py::array& arr) {
        return std::vector<size_t>(arr.shape(), arr.shape() + arr.ndim());
    });
    m.def("get_arr_shape", [](const py::array& arr, size_t dim) {
        return arr.shape(dim);
    });
    m.def("get_arr_strides", [](const py::array& arr) {
        return std::vector<size_t>(arr.strides(), arr.strides() + arr.ndim());
    });
    m.def("get_arr_strides", [](const py::array& arr, size_t dim) {
        return arr.strides(dim);
    });
    m.def("get_arr_writeable", [](const py::array& arr) {
        return arr.writeable();
    });
    m.def("get_arr_size", [](const py::array& arr) {
        return arr.size();
    });
    m.def("get_arr_itemsize", [](const py::array& arr) {
        return arr.itemsize();
    });
    m.def("get_arr_nbytes", [](const py::array& arr) {
        return arr.nbytes();
    });
    m.def("get_arr_owndata", [](const py::array& arr) {
        return arr.owndata();
    });
});
