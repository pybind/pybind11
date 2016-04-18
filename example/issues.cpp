/*
    example/issues.cpp -- collection of testcases for miscellaneous issues

    Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"

struct Base {
    virtual void dispatch(void) const = 0;
};

struct DispatchIssue : Base {
    virtual void dispatch(void) const {
        PYBIND11_OVERLOAD_PURE(void, Base, dispatch, /* no arguments */);
    }
};

void dispatch_issue_go(const Base * b) { b->dispatch(); }

PYBIND11_PLUGIN(mytest)
{
    pybind11::module m("mytest", "A test");


    return m.ptr();
}
void init_issues(py::module &m) {
    py::module m2 = m.def_submodule("issues");

    // #137: const char* isn't handled properly
    m2.def("print_cchar", [](const char *string) { std::cout << string << std::endl; });

    // #150: char bindings broken
    m2.def("print_char", [](char c) { std::cout << c << std::endl; });

    // #159: virtual function dispatch has problems with similar-named functions
    pybind11::class_<DispatchIssue> base(m2, "DispatchIssue");
    base.alias<Base>()
        .def(pybind11::init<>())
        .def("dispatch", &Base::dispatch);

    m2.def("dispatch_issue_go", &dispatch_issue_go);
}
