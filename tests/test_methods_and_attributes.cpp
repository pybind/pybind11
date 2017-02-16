/*
    tests/test_methods_and_attributes.cpp -- constructors, deconstructors, attribute access,
    __str__, argument and return value conventions

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include "constructor_stats.h"

class ExampleMandA {
public:
    ExampleMandA() { print_default_created(this); }
    ExampleMandA(int value) : value(value) { print_created(this, value); }
    ExampleMandA(const ExampleMandA &e) : value(e.value) { print_copy_created(this); }
    ExampleMandA(ExampleMandA &&e) : value(e.value) { print_move_created(this); }
    ~ExampleMandA() { print_destroyed(this); }

    std::string toString() {
        return "ExampleMandA[value=" + std::to_string(value) + "]";
    }

    void operator=(const ExampleMandA &e) { print_copy_assigned(this); value = e.value; }
    void operator=(ExampleMandA &&e) { print_move_assigned(this); value = e.value; }

    void add1(ExampleMandA other) { value += other.value; }           // passing by value
    void add2(ExampleMandA &other) { value += other.value; }          // passing by reference
    void add3(const ExampleMandA &other) { value += other.value; }    // passing by const reference
    void add4(ExampleMandA *other) { value += other->value; }         // passing by pointer
    void add5(const ExampleMandA *other) { value += other->value; }   // passing by const pointer

    void add6(int other) { value += other; }                      // passing by value
    void add7(int &other) { value += other; }                     // passing by reference
    void add8(const int &other) { value += other; }               // passing by const reference
    void add9(int *other) { value += *other; }                    // passing by pointer
    void add10(const int *other) { value += *other; }             // passing by const pointer

    ExampleMandA self1() { return *this; }                            // return by value
    ExampleMandA &self2() { return *this; }                           // return by reference
    const ExampleMandA &self3() { return *this; }                     // return by const reference
    ExampleMandA *self4() { return this; }                            // return by pointer
    const ExampleMandA *self5() { return this; }                      // return by const pointer

    int internal1() { return value; }                             // return by value
    int &internal2() { return value; }                            // return by reference
    const int &internal3() { return value; }                      // return by const reference
    int *internal4() { return &value; }                           // return by pointer
    const int *internal5() { return &value; }                     // return by const pointer

    py::str overloaded(int, float)   { return "(int, float)"; }
    py::str overloaded(float, int)   { return "(float, int)"; }
    py::str overloaded(int, int)     { return "(int, int)"; }
    py::str overloaded(float, float) { return "(float, float)"; }
    py::str overloaded(int, float)   const { return "(int, float) const"; }
    py::str overloaded(float, int)   const { return "(float, int) const"; }
    py::str overloaded(int, int)     const { return "(int, int) const"; }
    py::str overloaded(float, float) const { return "(float, float) const"; }

    int value = 0;
};

struct TestProperties {
    int value = 1;
    static int static_value;

    int get() const { return value; }
    void set(int v) { value = v; }

    static int static_get() { return static_value; }
    static void static_set(int v) { static_value = v; }
};

int TestProperties::static_value = 1;

struct SimpleValue { int value = 1; };

struct TestPropRVP {
    SimpleValue v1;
    SimpleValue v2;
    static SimpleValue sv1;
    static SimpleValue sv2;

    const SimpleValue &get1() const { return v1; }
    const SimpleValue &get2() const { return v2; }
    SimpleValue get_rvalue() const { return v2; }
    void set1(int v) { v1.value = v; }
    void set2(int v) { v2.value = v; }
};

SimpleValue TestPropRVP::sv1{};
SimpleValue TestPropRVP::sv2{};

class DynamicClass {
public:
    DynamicClass() { print_default_created(this); }
    ~DynamicClass() { print_destroyed(this); }
};

class CppDerivedDynamicClass : public DynamicClass { };

// py::arg/py::arg_v testing: these arguments just record their argument when invoked
class ArgInspector1 { public: std::string arg = "(default arg inspector 1)"; };
class ArgInspector2 { public: std::string arg = "(default arg inspector 2)"; };
class ArgAlwaysConverts { };
namespace pybind11 { namespace detail {
template <> struct type_caster<ArgInspector1> {
public:
    PYBIND11_TYPE_CASTER(ArgInspector1, _("ArgInspector1"));

    bool load(handle src, bool convert) {
        value.arg = "loading ArgInspector1 argument " +
            std::string(convert ? "WITH" : "WITHOUT") + " conversion allowed.  "
            "Argument value = " + (std::string) str(src);
        return true;
    }

    static handle cast(const ArgInspector1 &src, return_value_policy, handle) {
        return str(src.arg).release();
    }
};
template <> struct type_caster<ArgInspector2> {
public:
    PYBIND11_TYPE_CASTER(ArgInspector2, _("ArgInspector2"));

    bool load(handle src, bool convert) {
        value.arg = "loading ArgInspector2 argument " +
            std::string(convert ? "WITH" : "WITHOUT") + " conversion allowed.  "
            "Argument value = " + (std::string) str(src);
        return true;
    }

    static handle cast(const ArgInspector2 &src, return_value_policy, handle) {
        return str(src.arg).release();
    }
};
template <> struct type_caster<ArgAlwaysConverts> {
public:
    PYBIND11_TYPE_CASTER(ArgAlwaysConverts, _("ArgAlwaysConverts"));

    bool load(handle, bool convert) {
        return convert;
    }

    static handle cast(const ArgAlwaysConverts &, return_value_policy, handle) {
        return py::none();
    }
};
}}

/// Issue/PR #648: bad arg default debugging output
class NotRegistered {};

test_initializer methods_and_attributes([](py::module &m) {
    py::class_<ExampleMandA>(m, "ExampleMandA")
        .def(py::init<>())
        .def(py::init<int>())
        .def(py::init<const ExampleMandA&>())
        .def("add1", &ExampleMandA::add1)
        .def("add2", &ExampleMandA::add2)
        .def("add3", &ExampleMandA::add3)
        .def("add4", &ExampleMandA::add4)
        .def("add5", &ExampleMandA::add5)
        .def("add6", &ExampleMandA::add6)
        .def("add7", &ExampleMandA::add7)
        .def("add8", &ExampleMandA::add8)
        .def("add9", &ExampleMandA::add9)
        .def("add10", &ExampleMandA::add10)
        .def("self1", &ExampleMandA::self1)
        .def("self2", &ExampleMandA::self2)
        .def("self3", &ExampleMandA::self3)
        .def("self4", &ExampleMandA::self4)
        .def("self5", &ExampleMandA::self5)
        .def("internal1", &ExampleMandA::internal1)
        .def("internal2", &ExampleMandA::internal2)
        .def("internal3", &ExampleMandA::internal3)
        .def("internal4", &ExampleMandA::internal4)
        .def("internal5", &ExampleMandA::internal5)
#if defined(PYBIND11_OVERLOAD_CAST)
        .def("overloaded", py::overload_cast<int,   float>(&ExampleMandA::overloaded))
        .def("overloaded", py::overload_cast<float,   int>(&ExampleMandA::overloaded))
        .def("overloaded", py::overload_cast<int,     int>(&ExampleMandA::overloaded))
        .def("overloaded", py::overload_cast<float, float>(&ExampleMandA::overloaded))
        .def("overloaded_float", py::overload_cast<float, float>(&ExampleMandA::overloaded))
        .def("overloaded_const", py::overload_cast<int,   float>(&ExampleMandA::overloaded, py::const_))
        .def("overloaded_const", py::overload_cast<float,   int>(&ExampleMandA::overloaded, py::const_))
        .def("overloaded_const", py::overload_cast<int,     int>(&ExampleMandA::overloaded, py::const_))
        .def("overloaded_const", py::overload_cast<float, float>(&ExampleMandA::overloaded, py::const_))
#else
        .def("overloaded", static_cast<py::str (ExampleMandA::*)(int,   float)>(&ExampleMandA::overloaded))
        .def("overloaded", static_cast<py::str (ExampleMandA::*)(float,   int)>(&ExampleMandA::overloaded))
        .def("overloaded", static_cast<py::str (ExampleMandA::*)(int,     int)>(&ExampleMandA::overloaded))
        .def("overloaded", static_cast<py::str (ExampleMandA::*)(float, float)>(&ExampleMandA::overloaded))
        .def("overloaded_float", static_cast<py::str (ExampleMandA::*)(float, float)>(&ExampleMandA::overloaded))
        .def("overloaded_const", static_cast<py::str (ExampleMandA::*)(int,   float) const>(&ExampleMandA::overloaded))
        .def("overloaded_const", static_cast<py::str (ExampleMandA::*)(float,   int) const>(&ExampleMandA::overloaded))
        .def("overloaded_const", static_cast<py::str (ExampleMandA::*)(int,     int) const>(&ExampleMandA::overloaded))
        .def("overloaded_const", static_cast<py::str (ExampleMandA::*)(float, float) const>(&ExampleMandA::overloaded))
#endif
        .def("__str__", &ExampleMandA::toString)
        .def_readwrite("value", &ExampleMandA::value);

    py::class_<TestProperties>(m, "TestProperties")
        .def(py::init<>())
        .def_readonly("def_readonly", &TestProperties::value)
        .def_readwrite("def_readwrite", &TestProperties::value)
        .def_property_readonly("def_property_readonly", &TestProperties::get)
        .def_property("def_property", &TestProperties::get, &TestProperties::set)
        .def_readonly_static("def_readonly_static", &TestProperties::static_value)
        .def_readwrite_static("def_readwrite_static", &TestProperties::static_value)
        .def_property_readonly_static("def_property_readonly_static",
                                      [](py::object) { return TestProperties::static_get(); })
        .def_property_static("def_property_static",
                             [](py::object) { return TestProperties::static_get(); },
                             [](py::object, int v) { TestProperties::static_set(v); })
        .def_property_static("static_cls",
                             [](py::object cls) { return cls; },
                             [](py::object cls, py::function f) { f(cls); });

    py::class_<SimpleValue>(m, "SimpleValue")
        .def_readwrite("value", &SimpleValue::value);

    auto static_get1 = [](py::object) -> const SimpleValue & { return TestPropRVP::sv1; };
    auto static_get2 = [](py::object) -> const SimpleValue & { return TestPropRVP::sv2; };
    auto static_set1 = [](py::object, int v) { TestPropRVP::sv1.value = v; };
    auto static_set2 = [](py::object, int v) { TestPropRVP::sv2.value = v; };
    auto rvp_copy = py::return_value_policy::copy;

    py::class_<TestPropRVP>(m, "TestPropRVP")
        .def(py::init<>())
        .def_property_readonly("ro_ref", &TestPropRVP::get1)
        .def_property_readonly("ro_copy", &TestPropRVP::get2, rvp_copy)
        .def_property_readonly("ro_func", py::cpp_function(&TestPropRVP::get2, rvp_copy))
        .def_property("rw_ref", &TestPropRVP::get1, &TestPropRVP::set1)
        .def_property("rw_copy", &TestPropRVP::get2, &TestPropRVP::set2, rvp_copy)
        .def_property("rw_func", py::cpp_function(&TestPropRVP::get2, rvp_copy), &TestPropRVP::set2)
        .def_property_readonly_static("static_ro_ref", static_get1)
        .def_property_readonly_static("static_ro_copy", static_get2, rvp_copy)
        .def_property_readonly_static("static_ro_func", py::cpp_function(static_get2, rvp_copy))
        .def_property_static("static_rw_ref", static_get1, static_set1)
        .def_property_static("static_rw_copy", static_get2, static_set2, rvp_copy)
        .def_property_static("static_rw_func", py::cpp_function(static_get2, rvp_copy), static_set2)
        .def_property_readonly("rvalue", &TestPropRVP::get_rvalue)
        .def_property_readonly_static("static_rvalue", [](py::object) { return SimpleValue(); });

    struct MetaclassOverride { };
    py::class_<MetaclassOverride>(m, "MetaclassOverride", py::metaclass((PyObject *) &PyType_Type))
        .def_property_readonly_static("readonly", [](py::object) { return 1; });

#if !defined(PYPY_VERSION)
    py::class_<DynamicClass>(m, "DynamicClass", py::dynamic_attr())
        .def(py::init());

    py::class_<CppDerivedDynamicClass, DynamicClass>(m, "CppDerivedDynamicClass")
        .def(py::init());
#endif

    // Test converting.  The ArgAlwaysConverts is just there to make the first no-conversion pass
    // fail so that our call always ends up happening via the second dispatch (the one that allows
    // some conversion).
    class ArgInspector {
    public:
        ArgInspector1 f(ArgInspector1 a, ArgAlwaysConverts) { return a; }
        std::string g(ArgInspector1 a, const ArgInspector1 &b, int c, ArgInspector2 *d, ArgAlwaysConverts) {
            return a.arg + "\n" + b.arg + "\n" + std::to_string(c) + "\n" + d->arg;
        }
        static ArgInspector2 h(ArgInspector2 a, ArgAlwaysConverts) { return a; }
    };
    py::class_<ArgInspector>(m, "ArgInspector")
        .def(py::init<>())
        .def("f", &ArgInspector::f, py::arg(), py::arg() = ArgAlwaysConverts())
        .def("g", &ArgInspector::g, "a"_a.noconvert(), "b"_a, "c"_a.noconvert()=13, "d"_a=ArgInspector2(), py::arg() = ArgAlwaysConverts())
        .def_static("h", &ArgInspector::h, py::arg().noconvert(), py::arg() = ArgAlwaysConverts())
        ;
    m.def("arg_inspect_func", [](ArgInspector2 a, ArgInspector1 b, ArgAlwaysConverts) { return a.arg + "\n" + b.arg; },
            py::arg().noconvert(false), py::arg_v(nullptr, ArgInspector1()).noconvert(true), py::arg() = ArgAlwaysConverts());

    m.def("floats_preferred", [](double f) { return 0.5 * f; }, py::arg("f"));
    m.def("floats_only", [](double f) { return 0.5 * f; }, py::arg("f").noconvert());

    /// Issue/PR #648: bad arg default debugging output
#if !defined(NDEBUG)
    m.attr("debug_enabled") = true;
#else
    m.attr("debug_enabled") = false;
#endif
    m.def("bad_arg_def_named", []{
        auto m = py::module::import("pybind11_tests.issues");
        m.def("should_fail", [](int, NotRegistered) {}, py::arg(), py::arg("a") = NotRegistered());
    });
    m.def("bad_arg_def_unnamed", []{
        auto m = py::module::import("pybind11_tests.issues");
        m.def("should_fail", [](int, NotRegistered) {}, py::arg(), py::arg() = NotRegistered());
    });
});
