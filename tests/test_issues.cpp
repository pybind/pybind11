/*
    tests/test_issues.cpp -- collection of testcases for miscellaneous issues

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include "constructor_stats.h"
#include <pybind11/stl.h>
#include <pybind11/operators.h>

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

#define TRACKERS(CLASS) CLASS() { print_default_created(this); } ~CLASS() { print_destroyed(this); }
struct NestABase { int value = -2; TRACKERS(NestABase) };
struct NestA : NestABase { int value = 3; NestA& operator+=(int i) { value += i; return *this; } TRACKERS(NestA) };
struct NestB { NestA a; int value = 4; NestB& operator-=(int i) { value -= i; return *this; } TRACKERS(NestB) };
struct NestC { NestB b; int value = 5; NestC& operator*=(int i) { value *= i; return *this; } TRACKERS(NestC) };

void init_issues(py::module &m) {
    py::module m2 = m.def_submodule("issues");

#if !defined(_MSC_VER)
    // Visual Studio 2015 currently cannot compile this test
    // (see the comment in type_caster_base::make_copy_constructor)
    // #70 compilation issue if operator new is not public
    class NonConstructible { private: void *operator new(size_t bytes) throw(); };
    py::class_<NonConstructible>(m, "Foo");
    m2.def("getstmt", []() -> NonConstructible * { return nullptr; },
        py::return_value_policy::reference);
#endif

    // #137: const char* isn't handled properly
    m2.def("print_cchar", [](const char *s) { return std::string(s); });

    // #150: char bindings broken
    m2.def("print_char", [](char c) { return std::string(1, c); });

    // #159: virtual function dispatch has problems with similar-named functions
    struct Base { virtual std::string dispatch() const {
        /* for some reason MSVC2015 can't compile this if the function is pure virtual */
        return {};
    }; };

    struct DispatchIssue : Base {
        virtual std::string dispatch() const {
            PYBIND11_OVERLOAD_PURE(std::string, Base, dispatch, /* no arguments */);
        }
    };

    py::class_<Base, std::unique_ptr<Base>, DispatchIssue>(m2, "DispatchIssue")
        .def(py::init<>())
        .def("dispatch", &Base::dispatch);

    m2.def("dispatch_issue_go", [](const Base * b) { return b->dispatch(); });

    struct Placeholder { int i; Placeholder(int i) : i(i) { } };

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

    // #181: iterator passthrough did not compile
    m2.def("iterator_passthrough", [](py::iterator s) -> py::iterator {
        return py::make_iterator(std::begin(s), std::end(s));
    });

    // #187: issue involving std::shared_ptr<> return value policy & garbage collection
    struct ElementBase { virtual void foo() { } /* Force creation of virtual table */ };
    struct ElementA : ElementBase {
        ElementA(int v) : v(v) { }
        int value() { return v; }
        int v;
    };

    struct ElementList {
        void add(std::shared_ptr<ElementBase> e) { l.push_back(e); }
        std::vector<std::shared_ptr<ElementBase>> l;
    };

    py::class_<ElementBase, std::shared_ptr<ElementBase>> (m2, "ElementBase");

    py::class_<ElementA, std::shared_ptr<ElementA>>(m2, "ElementA", py::base<ElementBase>())
        .def(py::init<int>())
        .def("value", &ElementA::value);

    py::class_<ElementList, std::shared_ptr<ElementList>>(m2, "ElementList")
        .def(py::init<>())
        .def("add", &ElementList::add)
        .def("get", [](ElementList &el){
            py::list list;
            for (auto &e : el.l)
                list.append(py::cast(e));
            return list;
        });

    // (no id): should not be able to pass 'None' to a reference argument
    m2.def("get_element", [](ElementA &el) { return el.value(); });

    // (no id): don't cast doubles to ints
    m2.def("expect_float", [](float f) { return f; });
    m2.def("expect_int", [](int i) { return i; });

    // (no id): don't invoke Python dispatch code when instantiating C++
    // classes that were not extended on the Python side
    struct A {
        virtual ~A() {}
        virtual void f() { std::cout << "A.f()" << std::endl; }
    };

    struct PyA : A {
        PyA() { std::cout << "PyA.PyA()" << std::endl; }

        void f() override {
            std::cout << "PyA.f()" << std::endl;
            PYBIND11_OVERLOAD(void, A, f);
        }
    };

    auto call_f = [](A *a) { a->f(); };

    pybind11::class_<A, std::unique_ptr<A>, PyA>(m2, "A")
        .def(py::init<>())
        .def("f", &A::f);

    m2.def("call_f", call_f);

    try {
        py::class_<Placeholder>(m2, "Placeholder");
        throw std::logic_error("Expected an exception!");
    } catch (std::runtime_error &) {
        /* All good */
    }

    // Issue #283: __str__ called on uninitialized instance when constructor arguments invalid
    class StrIssue {
    public:
        StrIssue(int i) : val{i} {}
        StrIssue() : StrIssue(-1) {}
        int value() const { return val; }
    private:
        int val;
    };
    py::class_<StrIssue> si(m2, "StrIssue");
    si  .def(py::init<int>())
        .def(py::init<>())
        .def("__str__", [](const StrIssue &si) { return "StrIssue[" + std::to_string(si.value()) + "]"; })
        ;

    // Issue #328: first member in a class can't be used in operators
    py::class_<NestABase>(m2, "NestABase").def(py::init<>()).def_readwrite("value", &NestABase::value);
    py::class_<NestA>(m2, "NestA").def(py::init<>()).def(py::self += int())
        .def("as_base", [](NestA &a) -> NestABase& { return (NestABase&) a; }, py::return_value_policy::reference_internal);
    py::class_<NestB>(m2, "NestB").def(py::init<>()).def(py::self -= int()).def_readwrite("a", &NestB::a);
    py::class_<NestC>(m2, "NestC").def(py::init<>()).def(py::self *= int()).def_readwrite("b", &NestC::b);
    m2.def("get_NestA", [](const NestA &a) { return a.value; });
    m2.def("get_NestB", [](const NestB &b) { return b.value; });
    m2.def("get_NestC", [](const NestC &c) { return c.value; });
}
