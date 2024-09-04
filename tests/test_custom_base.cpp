/*
    tests/test_custom_base.cpp -- test custom type hierarchy support

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

namespace {

struct Base {
    int i = 5;
};

struct Derived {
    int j = 6;

    // just to prove the base can be anywhere
    Base base;
};

} // namespace

TEST_SUBMODULE(custom_base, m) {

    py::class_<Base>(m, "Base").def_readwrite("i", &Base::i);

    py::class_<Derived>(m, "Derived", py::custom_base<Base>([](void *o) -> void * {
                            return &reinterpret_cast<Derived *>(o)->base;
                        }))
        .def_readwrite("j", &Derived::j);

    m.def("create_derived", []() { return new Derived; });
    m.def("create_base", []() { return new Base; });

    m.def("base_i", [](Base *b) { return b->i; });

    m.def("derived_j", [](Derived *d) { return d->j; });
};
