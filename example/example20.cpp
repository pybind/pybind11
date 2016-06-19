/*
  example/example20.cpp -- Usage of structured numpy dtypes

  Copyright (c) 2016 Ivan Smirnov

  All rights reserved. Use of this source code is governed by a
  BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"

#include <pybind11/numpy.h>
#include <cstdint>
#include <iostream>

namespace py = pybind11;

struct Struct {
    bool x;
    uint32_t y;
    float z;
};

struct PackedStruct {
    bool x;
    uint32_t y;
    float z;
} __attribute__((packed));

struct NestedStruct {
    Struct a;
    PackedStruct b;
};

template <typename S>
py::array_t<S> create_recarray(size_t n) {
    auto arr = py::array(py::buffer_info(nullptr, sizeof(S),
                                         py::format_descriptor<S>::value(),
                                         1, { n }, { sizeof(S) }));
    auto buf = arr.request();
    auto ptr = static_cast<S*>(buf.ptr);
    for (size_t i = 0; i < n; i++) {
        ptr[i].x = i % 2;
        ptr[i].y = i;
        ptr[i].z = i * 1.5;
    }
    return arr;
}

void init_ex20(py::module &m) {
    PYBIND11_DTYPE(Struct, x, y, z);
    PYBIND11_DTYPE(PackedStruct, x, y, z);
    PYBIND11_DTYPE(NestedStruct, a, b);

    m.def("create_rec_simple", &create_recarray<Struct>);
    m.def("create_rec_packed", &create_recarray<PackedStruct>);
}
