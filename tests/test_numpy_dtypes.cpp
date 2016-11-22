/*
  tests/test_numpy_dtypes.cpp -- Structured and compound NumPy dtypes

  Copyright (c) 2016 Ivan Smirnov

  All rights reserved. Use of this source code is governed by a
  BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include <pybind11/numpy.h>

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
    uint64_t dummy2;
};

struct PartialNestedStruct {
    uint64_t dummy1;
    PartialStruct a;
    uint64_t dummy2;
};

struct UnboundStruct { };

struct StringStruct {
    char a[3];
    std::array<char, 3> b;
};

PYBIND11_PACKED(struct StructWithUglyNames {
    int8_t __x__;
    uint64_t __y__;
});

enum class E1 : int64_t { A = -1, B = 1 };
enum E2 : uint8_t { X = 1, Y = 2 };

PYBIND11_PACKED(struct EnumStruct {
    E1 e1;
    E2 e2;
});

std::ostream& operator<<(std::ostream& os, const StringStruct& v) {
    os << "a='";
    for (size_t i = 0; i < 3 && v.a[i]; i++) os << v.a[i];
    os << "',b='";
    for (size_t i = 0; i < 3 && v.b[i]; i++) os << v.b[i];
    return os << "'";
}

std::ostream& operator<<(std::ostream& os, const EnumStruct& v) {
    return os << "e1=" << (v.e1 == E1::A ? "A" : "B") << ",e2=" << (v.e2 == E2::X ? "X" : "Y");
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
        ptr[i].x = i % 2 != 0; ptr[i].y = (uint32_t) i; ptr[i].z = (float) i * 1.5f;
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
        ptr[i].a.x = i % 2 != 0; ptr[i].a.y = (uint32_t) i; ptr[i].a.z = (float) i * 1.5f;
        ptr[i].b.x = (i + 1) % 2 != 0; ptr[i].b.y = (uint32_t) (i + 1); ptr[i].b.z = (float) (i + 1) * 1.5f;
    }
    return arr;
}

py::array_t<PartialNestedStruct, 0> create_partial_nested(size_t n) {
    auto arr = mkarray_via_buffer<PartialNestedStruct>(n);
    auto req = arr.request();
    auto ptr = static_cast<PartialNestedStruct*>(req.ptr);
    for (size_t i = 0; i < n; i++) {
        ptr[i].a.x = i % 2 != 0; ptr[i].a.y = (uint32_t) i; ptr[i].a.z = (float) i * 1.5f;
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

py::array_t<EnumStruct, 0> create_enum_array(size_t n) {
    auto arr = mkarray_via_buffer<EnumStruct>(n);
    auto ptr = (EnumStruct *) arr.mutable_data();
    for (size_t i = 0; i < n; i++) {
        ptr[i].e1 = static_cast<E1>(-1 + ((int) i % 2) * 2);
        ptr[i].e2 = static_cast<E2>(1 + (i % 2));
    }
    return arr;
}

template <typename S>
py::list print_recarray(py::array_t<S, 0> arr) {
    const auto req = arr.request();
    const auto ptr = static_cast<S*>(req.ptr);
    auto l = py::list();
    for (size_t i = 0; i < req.size; i++) {
        std::stringstream ss;
        ss << ptr[i];
        l.append(py::str(ss.str()));
    }
    return l;
}

py::list print_format_descriptors() {
    const auto fmts = {
        py::format_descriptor<SimpleStruct>::format(),
        py::format_descriptor<PackedStruct>::format(),
        py::format_descriptor<NestedStruct>::format(),
        py::format_descriptor<PartialStruct>::format(),
        py::format_descriptor<PartialNestedStruct>::format(),
        py::format_descriptor<StringStruct>::format(),
        py::format_descriptor<EnumStruct>::format()
    };
    auto l = py::list();
    for (const auto &fmt : fmts) {
        l.append(py::cast(fmt));
    }
    return l;
}

py::list print_dtypes() {
    const auto dtypes = {
        py::str(py::dtype::of<SimpleStruct>()),
        py::str(py::dtype::of<PackedStruct>()),
        py::str(py::dtype::of<NestedStruct>()),
        py::str(py::dtype::of<PartialStruct>()),
        py::str(py::dtype::of<PartialNestedStruct>()),
        py::str(py::dtype::of<StringStruct>()),
        py::str(py::dtype::of<EnumStruct>()),
        py::str(py::dtype::of<StructWithUglyNames>())
    };
    auto l = py::list();
    for (const auto &s : dtypes) {
        l.append(s);
    }
    return l;
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
    py::buffer_info buf_ndim1_null(nullptr, 4, "i", 6);
    py::buffer_info buf_ndim2(vptr, 4, "i", 2, shape, strides);
    py::buffer_info buf_ndim2_null(nullptr, 4, "i", 2, shape, strides);

    auto fill = [](py::array arr) {
        auto req = arr.request();
        for (int i = 0; i < 6; i++) ((int32_t *) req.ptr)[i] = i + 1;
        return arr;
    };

    switch (i) {
    // shape: (3, 2)
    case 10: return arr_t(shape, strides, ptr);
    case 11: return py::array(shape, strides, ptr);
    case 12: return py::array(dtype, shape, strides, vptr);
    case 13: return arr_t(shape, ptr);
    case 14: return py::array(shape, ptr);
    case 15: return py::array(dtype, shape, vptr);
    case 16: return arr_t(buf_ndim2);
    case 17: return py::array(buf_ndim2);
    // shape: (3, 2) - post-fill
    case 20: return fill(arr_t(shape, strides));
    case 21: return py::array(shape, strides, ptr); // can't have nullptr due to templated ctor
    case 22: return fill(py::array(dtype, shape, strides));
    case 23: return fill(arr_t(shape));
    case 24: return py::array(shape, ptr); // can't have nullptr due to templated ctor
    case 25: return fill(py::array(dtype, shape));
    case 26: return fill(arr_t(buf_ndim2_null));
    case 27: return fill(py::array(buf_ndim2_null));
    // shape: (6, )
    case 30: return arr_t(6, ptr);
    case 31: return py::array(6, ptr);
    case 32: return py::array(dtype, 6, vptr);
    case 33: return arr_t(buf_ndim1);
    case 34: return py::array(buf_ndim1);
    // shape: (6, )
    case 40: return fill(arr_t(6));
    case 41: return py::array(6, ptr);  // can't have nullptr due to templated ctor
    case 42: return fill(py::array(dtype, 6));
    case 43: return fill(arr_t(buf_ndim1_null));
    case 44: return fill(py::array(buf_ndim1_null));
    }
    return arr_t();
}

py::list test_dtype_ctors() {
    py::list list;
    list.append(py::dtype("int32"));
    list.append(py::dtype(std::string("float64")));
    list.append(py::dtype::from_args(py::str("bool")));
    py::list names, offsets, formats;
    py::dict dict;
    names.append(py::str("a")); names.append(py::str("b")); dict["names"] = names;
    offsets.append(py::int_(1)); offsets.append(py::int_(10)); dict["offsets"] = offsets;
    formats.append(py::dtype("int32")); formats.append(py::dtype("float64")); dict["formats"] = formats;
    dict["itemsize"] = py::int_(20);
    list.append(py::dtype::from_args(dict));
    list.append(py::dtype(names, formats, offsets, 20));
    list.append(py::dtype(py::buffer_info((void *) 0, sizeof(unsigned int), "I", 1)));
    list.append(py::dtype(py::buffer_info((void *) 0, 0, "T{i:a:f:b:}", 1)));
    return list;
}

struct TrailingPaddingStruct {
    int32_t a;
    char b;
};

py::dtype trailing_padding_dtype() {
    return py::dtype::of<TrailingPaddingStruct>();
}

py::dtype buffer_to_dtype(py::buffer& buf) {
    return py::dtype(buf.request());
}

py::list test_dtype_methods() {
    py::list list;
    auto dt1 = py::dtype::of<int32_t>();
    auto dt2 = py::dtype::of<SimpleStruct>();
    list.append(dt1); list.append(dt2);
    list.append(py::bool_(dt1.has_fields())); list.append(py::bool_(dt2.has_fields()));
    list.append(py::int_(dt1.itemsize())); list.append(py::int_(dt2.itemsize()));
    return list;
}

test_initializer numpy_dtypes([](py::module &m) {
    try {
        py::module::import("numpy");
    } catch (...) {
        return;
    }

    // typeinfo may be registered before the dtype descriptor for scalar casts to work...
    py::class_<SimpleStruct>(m, "SimpleStruct");

    PYBIND11_NUMPY_DTYPE(SimpleStruct, x, y, z);
    PYBIND11_NUMPY_DTYPE(PackedStruct, x, y, z);
    PYBIND11_NUMPY_DTYPE(NestedStruct, a, b);
    PYBIND11_NUMPY_DTYPE(PartialStruct, x, y, z);
    PYBIND11_NUMPY_DTYPE(PartialNestedStruct, a);
    PYBIND11_NUMPY_DTYPE(StringStruct, a, b);
    PYBIND11_NUMPY_DTYPE(EnumStruct, e1, e2);
    PYBIND11_NUMPY_DTYPE(TrailingPaddingStruct, a, b);

    // ... or after
    py::class_<PackedStruct>(m, "PackedStruct");

    PYBIND11_NUMPY_DTYPE_EX(StructWithUglyNames, __x__, "x", __y__, "y");

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
    m.def("create_enum_array", &create_enum_array);
    m.def("print_enum_array", &print_recarray<EnumStruct>);
    m.def("test_array_ctors", &test_array_ctors);
    m.def("test_dtype_ctors", &test_dtype_ctors);
    m.def("test_dtype_methods", &test_dtype_methods);
    m.def("trailing_padding_dtype", &trailing_padding_dtype);
    m.def("buffer_to_dtype", &buffer_to_dtype);
    m.def("f_simple", [](SimpleStruct s) { return s.y * 10; });
    m.def("f_packed", [](PackedStruct s) { return s.y * 10; });
    m.def("f_nested", [](NestedStruct s) { return s.a.y * 10; });
    m.def("register_dtype", []() { PYBIND11_NUMPY_DTYPE(SimpleStruct, x, y, z); });
});

#undef PYBIND11_PACKED
