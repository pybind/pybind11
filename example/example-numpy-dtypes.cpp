/*
  example/example-numpy-dtypes.cpp -- Structured and compound NumPy dtypes

  Copyright (c) 2016 Ivan Smirnov

  All rights reserved. Use of this source code is governed by a
  BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"

#include <pybind11/numpy.h>
#include <cstdint>
#include <iostream>

#ifdef __GNUC__
#define PYBIND11_PACKED(cls) cls __attribute__((__packed__))
#else
#define PYBIND11_PACKED(cls) __pragma(pack(push, 1)) cls __pragma(pack(pop))
#endif

namespace py = pybind11;

struct SimpleStruct {
    bool x;
    uint32_t y;
    float z;
};

std::ostream& operator<<(std::ostream& os, const SimpleStruct& v) {
    return os << "s:" << v.x << "," << v.y << "," << v.z;
}

PYBIND11_PACKED(struct PackedStruct {
    bool x;
    uint32_t y;
    float z;
});

std::ostream& operator<<(std::ostream& os, const PackedStruct& v) {
    return os << "p:" << v.x << "," << v.y << "," << v.z;
}

PYBIND11_PACKED(struct NestedStruct {
    SimpleStruct a;
    PackedStruct b;
});

std::ostream& operator<<(std::ostream& os, const NestedStruct& v) {
    return os << "n:a=" << v.a << ";b=" << v.b;
}

struct PartialStruct {
    bool x;
    uint32_t y;
    float z;
    long dummy2;
};

struct PartialNestedStruct {
    long dummy1;
    PartialStruct a;
    long dummy2;
};

struct UnboundStruct { };

struct StringStruct {
    char a[3];
    std::array<char, 3> b;
};

std::ostream& operator<<(std::ostream& os, const StringStruct& v) {
    os << "a='";
    for (size_t i = 0; i < 3 && v.a[i]; i++) os << v.a[i];
    os << "',b='";
    for (size_t i = 0; i < 3 && v.b[i]; i++) os << v.b[i];
    return os << "'";
}

template <typename T>
py::array mkarray_via_buffer(size_t n) {
    return py::array(py::buffer_info(nullptr, sizeof(T),
                                     py::format_descriptor<T>::format(),
                                     1, { n }, { sizeof(T) }));
}

template <typename S>
py::array_t<S, 0> create_recarray(size_t n) {
    auto arr = mkarray_via_buffer<S>(n);
    auto req = arr.request();
    auto ptr = static_cast<S*>(req.ptr);
    for (size_t i = 0; i < n; i++) {
        ptr[i].x = i % 2; ptr[i].y = (uint32_t) i; ptr[i].z = (float) i * 1.5f;
    }
    return arr;
}

std::string get_format_unbound() {
    return py::format_descriptor<UnboundStruct>::format();
}

py::array_t<NestedStruct, 0> create_nested(size_t n) {
    auto arr = mkarray_via_buffer<NestedStruct>(n);
    auto req = arr.request();
    auto ptr = static_cast<NestedStruct*>(req.ptr);
    for (size_t i = 0; i < n; i++) {
        ptr[i].a.x = i % 2; ptr[i].a.y = (uint32_t) i; ptr[i].a.z = (float) i * 1.5f;
        ptr[i].b.x = (i + 1) % 2; ptr[i].b.y = (uint32_t) (i + 1); ptr[i].b.z = (float) (i + 1) * 1.5f;
    }
    return arr;
}

py::array_t<PartialNestedStruct, 0> create_partial_nested(size_t n) {
    auto arr = mkarray_via_buffer<PartialNestedStruct>(n);
    auto req = arr.request();
    auto ptr = static_cast<PartialNestedStruct*>(req.ptr);
    for (size_t i = 0; i < n; i++) {
        ptr[i].a.x = i % 2; ptr[i].a.y = (uint32_t) i; ptr[i].a.z = (float) i * 1.5f;
    }
    return arr;
}

py::array_t<StringStruct, 0> create_string_array(bool non_empty) {
    auto arr = mkarray_via_buffer<StringStruct>(non_empty ? 4 : 0);
    if (non_empty) {
        auto req = arr.request();
        auto ptr = static_cast<StringStruct*>(req.ptr);
        for (size_t i = 0; i < req.size * req.itemsize; i++)
            static_cast<char*>(req.ptr)[i] = 0;
        ptr[1].a[0] = 'a'; ptr[1].b[0] = 'a';
        ptr[2].a[0] = 'a'; ptr[2].b[0] = 'a';
        ptr[3].a[0] = 'a'; ptr[3].b[0] = 'a';

        ptr[2].a[1] = 'b'; ptr[2].b[1] = 'b';
        ptr[3].a[1] = 'b'; ptr[3].b[1] = 'b';

        ptr[3].a[2] = 'c'; ptr[3].b[2] = 'c';
    }
    return arr;
}

template <typename S>
void print_recarray(py::array_t<S, 0> arr) {
    auto req = arr.request();
    auto ptr = static_cast<S*>(req.ptr);
    for (size_t i = 0; i < req.size; i++)
        std::cout << ptr[i] << std::endl;
}

void print_format_descriptors() {
    std::cout << py::format_descriptor<SimpleStruct>::format() << std::endl;
    std::cout << py::format_descriptor<PackedStruct>::format() << std::endl;
    std::cout << py::format_descriptor<NestedStruct>::format() << std::endl;
    std::cout << py::format_descriptor<PartialStruct>::format() << std::endl;
    std::cout << py::format_descriptor<PartialNestedStruct>::format() << std::endl;
    std::cout << py::format_descriptor<StringStruct>::format() << std::endl;
}

void print_dtypes() {
    std::cout << (std::string) py::dtype::of<SimpleStruct>().str() << std::endl;
    std::cout << (std::string) py::dtype::of<PackedStruct>().str() << std::endl;
    std::cout << (std::string) py::dtype::of<NestedStruct>().str() << std::endl;
    std::cout << (std::string) py::dtype::of<PartialStruct>().str() << std::endl;
    std::cout << (std::string) py::dtype::of<PartialNestedStruct>().str() << std::endl;
    std::cout << (std::string) py::dtype::of<StringStruct>().str() << std::endl;
}

py::array_t<int32_t, 0> test_array_ctors(int i) {
    using arr_t = py::array_t<int32_t, 0>;

    std::vector<int32_t> data { 1, 2, 3, 4, 5, 6 };
    std::vector<size_t> shape { 3, 2 };
    std::vector<size_t> strides { 8, 4 };

    auto ptr = data.data();
    auto vptr = (void *) ptr;
    auto dtype = py::dtype("int32");

    py::buffer_info buf_ndim1(vptr, 4, "i", 6);
    py::buffer_info buf_ndim2(vptr, 4, "i", 2, shape, strides);

    switch (i) {
    // shape: (3, 2)
    case 0: return arr_t(shape, ptr, strides);
    case 1: return py::array(shape, ptr, strides);
    case 2: return py::array(dtype, shape, vptr, strides);
    case 3: return arr_t(shape, ptr);
    case 4: return py::array(shape, ptr);
    case 5: return py::array(dtype, shape, vptr);
    case 6: return arr_t(buf_ndim2);
    case 7: return py::array(buf_ndim2);
    // shape: (6, )
    case 8: return arr_t(6, ptr);
    case 9: return py::array(6, ptr);
    case 10: return py::array(dtype, 6, vptr);
    case 11: return arr_t(buf_ndim1);
    case 12: return py::array(buf_ndim1);
    }
    return arr_t();
}

void init_ex_numpy_dtypes(py::module &m) {
    PYBIND11_NUMPY_DTYPE(SimpleStruct, x, y, z);
    PYBIND11_NUMPY_DTYPE(PackedStruct, x, y, z);
    PYBIND11_NUMPY_DTYPE(NestedStruct, a, b);
    PYBIND11_NUMPY_DTYPE(PartialStruct, x, y, z);
    PYBIND11_NUMPY_DTYPE(PartialNestedStruct, a);
    PYBIND11_NUMPY_DTYPE(StringStruct, a, b);

    m.def("create_rec_simple", &create_recarray<SimpleStruct>);
    m.def("create_rec_packed", &create_recarray<PackedStruct>);
    m.def("create_rec_nested", &create_nested);
    m.def("create_rec_partial", &create_recarray<PartialStruct>);
    m.def("create_rec_partial_nested", &create_partial_nested);
    m.def("print_format_descriptors", &print_format_descriptors);
    m.def("print_rec_simple", &print_recarray<SimpleStruct>);
    m.def("print_rec_packed", &print_recarray<PackedStruct>);
    m.def("print_rec_nested", &print_recarray<NestedStruct>);
    m.def("print_dtypes", &print_dtypes);
    m.def("get_format_unbound", &get_format_unbound);
    m.def("create_string_array", &create_string_array);
    m.def("print_string_array", &print_recarray<StringStruct>);
    m.def("test_array_ctors", &test_array_ctors);
}

#undef PYBIND11_PACKED
