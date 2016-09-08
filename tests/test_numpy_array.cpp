/*
    tests/test_numpy_array.cpp -- test core array functionality

    Copyright (c) 2016 Ivan Smirnov <i.s.smirnov@gmail.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <vector>

using arr = py::array;
using arr_t = py::array_t<uint16_t, 0>;

template<typename... Ix> arr data(const arr& a, Ix&&... index) {
    return arr(a.nbytes() - a.offset_at(index...), (const uint8_t *) a.data(index...));
}

template<typename... Ix> arr data_t(const arr_t& a, Ix&&... index) {
    return arr(a.size() - a.index_at(index...), a.data(index...));
}

arr& mutate_data(arr& a) {
    auto ptr = (uint8_t *) a.mutable_data();
    for (size_t i = 0; i < a.nbytes(); i++)
        ptr[i] = (uint8_t) (ptr[i] * 2);
    return a;
}

arr_t& mutate_data_t(arr_t& a) {
    auto ptr = a.mutable_data();
    for (size_t i = 0; i < a.size(); i++)
        ptr[i]++;
    return a;
}

template<typename... Ix> arr& mutate_data(arr& a, Ix&&... index) {
    auto ptr = (uint8_t *) a.mutable_data(index...);
    for (size_t i = 0; i < a.nbytes() - a.offset_at(index...); i++)
        ptr[i] = (uint8_t) (ptr[i] * 2);
    return a;
}

template<typename... Ix> arr_t& mutate_data_t(arr_t& a, Ix&&... index) {
    auto ptr = a.mutable_data(index...);
    for (size_t i = 0; i < a.size() - a.index_at(index...); i++)
        ptr[i]++;
    return a;
}

template<typename... Ix> size_t index_at(const arr& a, Ix&&... idx) { return a.index_at(idx...); }
template<typename... Ix> size_t index_at_t(const arr_t& a, Ix&&... idx) { return a.index_at(idx...); }
template<typename... Ix> size_t offset_at(const arr& a, Ix&&... idx) { return a.offset_at(idx...); }
template<typename... Ix> size_t offset_at_t(const arr_t& a, Ix&&... idx) { return a.offset_at(idx...); }
template<typename... Ix> size_t at_t(const arr_t& a, Ix&&... idx) { return a.at(idx...); }
template<typename... Ix> arr_t& mutate_at_t(arr_t& a, Ix&&... idx) { a.mutable_at(idx...)++; return a; }

#define def_index_fn(name, type) \
    sm.def(#name, [](type a) { return name(a); }); \
    sm.def(#name, [](type a, int i) { return name(a, i); }); \
    sm.def(#name, [](type a, int i, int j) { return name(a, i, j); }); \
    sm.def(#name, [](type a, int i, int j, int k) { return name(a, i, j, k); });

test_initializer numpy_array([](py::module &m) {
    auto sm = m.def_submodule("array");

    sm.def("ndim", [](const arr& a) { return a.ndim(); });
    sm.def("shape", [](const arr& a) { return arr(a.ndim(), a.shape()); });
    sm.def("shape", [](const arr& a, size_t dim) { return a.shape(dim); });
    sm.def("strides", [](const arr& a) { return arr(a.ndim(), a.strides()); });
    sm.def("strides", [](const arr& a, size_t dim) { return a.strides(dim); });
    sm.def("writeable", [](const arr& a) { return a.writeable(); });
    sm.def("size", [](const arr& a) { return a.size(); });
    sm.def("itemsize", [](const arr& a) { return a.itemsize(); });
    sm.def("nbytes", [](const arr& a) { return a.nbytes(); });
    sm.def("owndata", [](const arr& a) { return a.owndata(); });

    def_index_fn(data, const arr&);
    def_index_fn(data_t, const arr_t&);
    def_index_fn(index_at, const arr&);
    def_index_fn(index_at_t, const arr_t&);
    def_index_fn(offset_at, const arr&);
    def_index_fn(offset_at_t, const arr_t&);
    def_index_fn(mutate_data, arr&);
    def_index_fn(mutate_data_t, arr_t&);
    def_index_fn(at_t, const arr_t&);
    def_index_fn(mutate_at_t, arr_t&);
});
