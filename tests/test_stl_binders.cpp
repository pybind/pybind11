/*
    tests/test_stl_binders.cpp -- Usage of stl_binders functions

    Copyright (c) 2016 Sergey Lyskov

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

#include <pybind11/stl_bind.h>
#include <map>
#include <deque>
#include <unordered_map>

#ifdef _MSC_VER
// We get some really long type names here which causes MSVC to emit warnings
#  pragma warning(disable: 4503) // warning C4503: decorated name length exceeded, name was truncated
#endif

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

/// Issue #487: binding std::vector<E> with E non-copyable
class E_nc {
public:
    explicit E_nc(int i) : value{i} {}
    E_nc(const E_nc &) = delete;
    E_nc &operator=(const E_nc &) = delete;
    E_nc(E_nc &&) = default;
    E_nc &operator=(E_nc &&) = default;

    int value;
};

template <class Container> Container *one_to_n(int n) {
    auto v = new Container();
    for (int i = 1; i <= n; i++)
        v->emplace_back(i);
    return v;
}

template <class Map> Map *times_ten(int n) {
    auto m = new Map();
    for (int i = 1; i <= n; i++)
        m->emplace(int(i), E_nc(10*i));
    return m;
}

test_initializer stl_binder_vector([](py::module &m) {
    py::class_<El>(m, "El")
        .def(py::init<int>());

    py::bind_vector<std::vector<unsigned int>>(m, "VectorInt");
    py::bind_vector<std::vector<bool>>(m, "VectorBool");

    py::bind_vector<std::vector<El>>(m, "VectorEl");

    py::bind_vector<std::vector<std::vector<El>>>(m, "VectorVectorEl");

});

test_initializer stl_binder_map([](py::module &m) {
    py::bind_map<std::map<std::string, double>>(m, "MapStringDouble");
    py::bind_map<std::unordered_map<std::string, double>>(m, "UnorderedMapStringDouble");

    py::bind_map<std::map<std::string, double const>>(m, "MapStringDoubleConst");
    py::bind_map<std::unordered_map<std::string, double const>>(m, "UnorderedMapStringDoubleConst");

});

test_initializer stl_binder_noncopyable([](py::module &m) {
    py::class_<E_nc>(m, "ENC")
        .def(py::init<int>())
        .def_readwrite("value", &E_nc::value);

    py::bind_vector<std::vector<E_nc>>(m, "VectorENC");
    m.def("get_vnc", &one_to_n<std::vector<E_nc>>, py::return_value_policy::reference);

    py::bind_vector<std::deque<E_nc>>(m, "DequeENC");
    m.def("get_dnc", &one_to_n<std::deque<E_nc>>, py::return_value_policy::reference);

    py::bind_map<std::map<int, E_nc>>(m, "MapENC");
    m.def("get_mnc", &times_ten<std::map<int, E_nc>>, py::return_value_policy::reference);

    py::bind_map<std::unordered_map<int, E_nc>>(m, "UmapENC");
    m.def("get_umnc", &times_ten<std::unordered_map<int, E_nc>>, py::return_value_policy::reference);
});

