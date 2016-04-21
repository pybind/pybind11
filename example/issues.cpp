/*
    example/issues.cpp -- collection of testcases for miscellaneous issues

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"
#include <pybind11/stl.h>

struct Base {
    virtual void dispatch(void) const = 0;
};

struct DispatchIssue : Base {
    virtual void dispatch(void) const {
        PYBIND11_OVERLOAD_PURE(void, Base, dispatch, /* no arguments */);
    }
};

struct Placeholder { int i; Placeholder(int i) : i(i) { } };

void dispatch_issue_go(const Base * b) { b->dispatch(); }

void init_issues(py::module &m) {
    py::module m2 = m.def_submodule("issues");

    // #137: const char* isn't handled properly
    m2.def("print_cchar", [](const char *string) { std::cout << string << std::endl; });

    // #150: char bindings broken
    m2.def("print_char", [](char c) { std::cout << c << std::endl; });

    // #159: virtual function dispatch has problems with similar-named functions
    py::class_<DispatchIssue> base(m2, "DispatchIssue");
    base.alias<Base>()
        .def(py::init<>())
        .def("dispatch", &Base::dispatch);

    m2.def("dispatch_issue_go", &dispatch_issue_go);

    py::class_<Placeholder>(m2, "Placeholder")
        .def(py::init<int>())
        .def("__repr__", [](const Placeholder &p) { return "Placeholder[" + std::to_string(p.i) + "]"; });

    // #171: Can't return reference wrappers (or STL datastructures containing them)
    m2.def("return_vec_of_reference_wrapper", [](std::reference_wrapper<Placeholder> p4){
        Placeholder *p1 = new Placeholder{1};
        Placeholder *p2 = new Placeholder{2};
        Placeholder *p3 = new Placeholder{3};
        std::vector<std::reference_wrapper<Placeholder>> v;
        v.push_back(std::ref(*p1));
        v.push_back(std::ref(*p2));
        v.push_back(std::ref(*p3));
        v.push_back(p4);
        return v;
    });
}
