/*
    tests/test_multiple_inheritance.cpp -- multiple inheritance,
    implicit MI casts

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

struct Base1 {
    Base1(int i) : i(i) { }
    int foo() { return i; }
    int i;
};

struct Base2 {
    Base2(int i) : i(i) { }
    int bar() { return i; }
    int i;
};

struct Base12 : Base1, Base2 {
    Base12(int i, int j) : Base1(i), Base2(j) { }
};

struct MIType : Base12 {
    MIType(int i, int j) : Base12(i, j) { }
};

test_initializer multiple_inheritance([](py::module &m) {
    py::class_<Base1> b1(m, "Base1");
    b1.def(py::init<int>())
      .def("foo", &Base1::foo);

    py::class_<Base2> b2(m, "Base2");
    b2.def(py::init<int>())
      .def("bar", &Base2::bar);

    py::class_<Base12, Base1, Base2>(m, "Base12");

    py::class_<MIType, Base12>(m, "MIType")
        .def(py::init<int, int>());

    // Uncommenting this should result in a compile time failure (MI can only be specified via
    // template parameters because pybind has to know the types involved; see discussion in #742 for
    // details).
//    struct Base12v2 : Base1, Base2 {
//        Base12v2(int i, int j) : Base1(i), Base2(j) { }
//    };
//    py::class_<Base12v2>(m, "Base12v2", b1, b2)
//        .def(py::init<int, int>());
});

/* Test the case where not all base classes are specified,
   and where pybind11 requires the py::multiple_inheritance
   flag to perform proper casting between types */

struct Base1a {
    Base1a(int i) : i(i) { }
    int foo() { return i; }
    int i;
};

struct Base2a {
    Base2a(int i) : i(i) { }
    int bar() { return i; }
    int i;
};

struct Base12a : Base1a, Base2a {
    Base12a(int i, int j) : Base1a(i), Base2a(j) { }
};

test_initializer multiple_inheritance_nonexplicit([](py::module &m) {
    py::class_<Base1a, std::shared_ptr<Base1a>>(m, "Base1a")
        .def(py::init<int>())
        .def("foo", &Base1a::foo);

    py::class_<Base2a, std::shared_ptr<Base2a>>(m, "Base2a")
        .def(py::init<int>())
        .def("bar", &Base2a::bar);

    py::class_<Base12a, /* Base1 missing */ Base2a,
               std::shared_ptr<Base12a>>(m, "Base12a", py::multiple_inheritance())
        .def(py::init<int, int>());

    m.def("bar_base2a", [](Base2a *b) { return b->bar(); });
    m.def("bar_base2a_sharedptr", [](std::shared_ptr<Base2a> b) { return b->bar(); });
});

// Issue #801: invalid casting to derived type with MI bases
struct I801B1 { int a = 1; virtual ~I801B1() = default; };
struct I801B2 { int b = 2; virtual ~I801B2() = default; };
struct I801C : I801B1, I801B2 {};
struct I801D : I801C {}; // Indirect MI

// one more example for issue #801
class objbase : public std::enable_shared_from_this<objbase> {
    int a = 1;
public:
    virtual std::string name() const {
        return "objbase";
    }
};
using sp_obj = std::shared_ptr<objbase>;
using sp_cobj = std::shared_ptr<const objbase>;

class iface {
    int b = 2;
public:
    virtual int f() const = 0;
};

class mytype : public iface, public objbase {
public:
    std::string name() const override { return "mytype"; }
    int f() const override { return 42; }
};
using sp_my = std::shared_ptr<mytype>;

class myslot {
public:
    virtual void act(sp_cobj param) const = 0;
};

// trmapoline
class py_myslot : public myslot {
public:
    using myslot::myslot;

    void act(sp_cobj obj) const override {
        PYBIND11_OVERLOAD_PURE(void, myslot, act, std::move(obj));
    }
};

test_initializer multiple_inheritance_casting([](py::module &m) {
    py::class_<I801B1, std::shared_ptr<I801B1>>(m, "I801B1").def(py::init<>()).def_readonly("a", &I801B1::a);
    py::class_<I801B2, std::shared_ptr<I801B2>>(m, "I801B2").def(py::init<>()).def_readonly("b", &I801B2::b);
    py::class_<I801C, I801B1, I801B2, std::shared_ptr<I801C>>(m, "I801C").def(py::init<>());
    py::class_<I801D, I801C, std::shared_ptr<I801D>>(m, "I801D").def(py::init<>());

    // Two separate issues here: first, we want to recognize a pointer to a base type as being a
    // known instance even when the pointer value is unequal (i.e. due to a non-first
    // multiple-inheritance base class):
    m.def("i801b1_c", [](I801C *c) { return static_cast<I801B1 *>(c); });
    m.def("i801b2_c", [](I801C *c) { return static_cast<I801B2 *>(c); });
    m.def("i801b1_d", [](I801D *d) { return static_cast<I801B1 *>(d); });
    m.def("i801b2_d", [](I801D *d) { return static_cast<I801B2 *>(d); });

    // Second, when returned a base class pointer to a derived instance, we cannot assume that the
    // pointer is `reinterpret_cast`able to the derived pointer because, like above, the base class
    // pointer could be offset.
    m.def("i801c_b1", []() -> I801B1 * { return new I801C(); });
    m.def("i801c_b2", []() -> I801B2 * { return new I801C(); });
    m.def("i801d_b1", []() -> I801B1 * { return new I801D(); });
    m.def("i801d_b2", []() -> I801B2 * { return new I801D(); });

    // test one more example
    py::class_<objbase, sp_obj>(m, "objbase")
        .def(py::init<>())
        .def("name", &objbase::name)
        .def_property_readonly("refs", [](const objbase& src) { return src.shared_from_this().use_count() - 1; })
        ;
    py::class_<iface, std::shared_ptr<iface>>(m, "iface")
        .def("f", &iface::f)
        ;
    py::class_<mytype, iface, objbase, sp_my>(m, "mytype")
        .def(py::init<>())
        .def("test_slot", [](mytype& src, std::shared_ptr<myslot> s) {
            s->act(std::dynamic_pointer_cast<mytype>(src.shared_from_this()));
        })
        ;
    py::class_<myslot, py_myslot, std::shared_ptr<myslot>>(m, "myslot")
        .def(py::init_alias<>())
        .def("act", &myslot::act)
        ;

    m.def("who_am_i", [](sp_cobj inst) {
        return std::string("My name is ") + inst->name();
    });

});

struct Vanilla {
    std::string vanilla() { return "Vanilla"; };
};

struct WithStatic1 {
    static std::string static_func1() { return "WithStatic1"; };
    static int static_value1;
};

struct WithStatic2 {
    static std::string static_func2() { return "WithStatic2"; };
    static int static_value2;
};

struct WithDict { };

struct VanillaStaticMix1 : Vanilla, WithStatic1, WithStatic2 {
    static std::string static_func() { return "VanillaStaticMix1"; }
    static int static_value;
};

struct VanillaStaticMix2 : WithStatic1, Vanilla, WithStatic2 {
    static std::string static_func() { return "VanillaStaticMix2"; }
    static int static_value;
};

struct VanillaDictMix1 : Vanilla, WithDict { };
struct VanillaDictMix2 : WithDict, Vanilla { };

int WithStatic1::static_value1 = 1;
int WithStatic2::static_value2 = 2;
int VanillaStaticMix1::static_value = 12;
int VanillaStaticMix2::static_value = 12;

test_initializer mi_static_properties([](py::module &pm) {
    auto m = pm.def_submodule("mi");

    py::class_<Vanilla>(m, "Vanilla")
        .def(py::init<>())
        .def("vanilla", &Vanilla::vanilla);

    py::class_<WithStatic1>(m, "WithStatic1")
        .def(py::init<>())
        .def_static("static_func1", &WithStatic1::static_func1)
        .def_readwrite_static("static_value1", &WithStatic1::static_value1);

    py::class_<WithStatic2>(m, "WithStatic2")
        .def(py::init<>())
        .def_static("static_func2", &WithStatic2::static_func2)
        .def_readwrite_static("static_value2", &WithStatic2::static_value2);

    py::class_<VanillaStaticMix1, Vanilla, WithStatic1, WithStatic2>(
        m, "VanillaStaticMix1")
        .def(py::init<>())
        .def_static("static_func", &VanillaStaticMix1::static_func)
        .def_readwrite_static("static_value", &VanillaStaticMix1::static_value);

    py::class_<VanillaStaticMix2, WithStatic1, Vanilla, WithStatic2>(
        m, "VanillaStaticMix2")
        .def(py::init<>())
        .def_static("static_func", &VanillaStaticMix2::static_func)
        .def_readwrite_static("static_value", &VanillaStaticMix2::static_value);

#if !defined(PYPY_VERSION)
    py::class_<WithDict>(m, "WithDict", py::dynamic_attr()).def(py::init<>());
    py::class_<VanillaDictMix1, Vanilla, WithDict>(m, "VanillaDictMix1").def(py::init<>());
    py::class_<VanillaDictMix2, WithDict, Vanilla>(m, "VanillaDictMix2").def(py::init<>());
#endif
});
