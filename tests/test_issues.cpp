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
#include <pybind11/complex.h>

#define TRACKERS(CLASS) CLASS() { print_default_created(this); } ~CLASS() { print_destroyed(this); }
struct NestABase { int value = -2; TRACKERS(NestABase) };
struct NestA : NestABase { int value = 3; NestA& operator+=(int i) { value += i; return *this; } TRACKERS(NestA) };
struct NestB { NestA a; int value = 4; NestB& operator-=(int i) { value -= i; return *this; } TRACKERS(NestB) };
struct NestC { NestB b; int value = 5; NestC& operator*=(int i) { value *= i; return *this; } TRACKERS(NestC) };

/// #393
class OpTest1 {};
class OpTest2 {};

OpTest1 operator+(const OpTest1 &, const OpTest1 &) {
    py::print("Add OpTest1 with OpTest1");
    return OpTest1();
}
OpTest2 operator+(const OpTest2 &, const OpTest2 &) {
    py::print("Add OpTest2 with OpTest2");
    return OpTest2();
}
OpTest2 operator+(const OpTest2 &, const OpTest1 &) {
    py::print("Add OpTest2 with OpTest1");
    return OpTest2();
}

// #461
class Dupe1 {
public:
    Dupe1(int v) : v_{v} {}
    int get_value() const { return v_; }
private:
    int v_;
};
class Dupe2 {};
class Dupe3 {};
class DupeException : public std::runtime_error {};

// #478
template <typename T> class custom_unique_ptr {
public:
    custom_unique_ptr() { print_default_created(this); }
    custom_unique_ptr(T *ptr) : _ptr{ptr} { print_created(this, ptr); }
    custom_unique_ptr(custom_unique_ptr<T> &&move) : _ptr{move._ptr} { move._ptr = nullptr; print_move_created(this); }
    custom_unique_ptr &operator=(custom_unique_ptr<T> &&move) { print_move_assigned(this); if (_ptr) destruct_ptr(); _ptr = move._ptr; move._ptr = nullptr; return *this; }
    custom_unique_ptr(const custom_unique_ptr<T> &) = delete;
    void operator=(const custom_unique_ptr<T> &copy) = delete;
    ~custom_unique_ptr() { print_destroyed(this); if (_ptr) destruct_ptr(); }
private:
    T *_ptr = nullptr;
    void destruct_ptr() { delete _ptr; }
};
PYBIND11_DECLARE_HOLDER_TYPE(T, custom_unique_ptr<T>);

/// Issue #528: templated constructor
struct TplConstrClass {
    template <typename T> TplConstrClass(const T &arg) : str{arg} {}
    std::string str;
    bool operator==(const TplConstrClass &t) const { return t.str == str; }
};
namespace std {
template <> struct hash<TplConstrClass> { size_t operator()(const TplConstrClass &t) const { return std::hash<std::string>()(t.str); } };
}


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

    py::class_<Base, DispatchIssue>(m2, "DispatchIssue")
        .def(py::init<>())
        .def("dispatch", &Base::dispatch);

    m2.def("dispatch_issue_go", [](const Base * b) { return b->dispatch(); });

    struct Placeholder { int i; Placeholder(int i) : i(i) { } };

    py::class_<Placeholder>(m2, "Placeholder")
        .def(py::init<int>())
        .def("__repr__", [](const Placeholder &p) { return "Placeholder[" + std::to_string(p.i) + "]"; });

    // #171: Can't return reference wrappers (or STL datastructures containing them)
    m2.def("return_vec_of_reference_wrapper", [](std::reference_wrapper<Placeholder> p4) {
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

    py::class_<ElementA, ElementBase, std::shared_ptr<ElementA>>(m2, "ElementA")
        .def(py::init<int>())
        .def("value", &ElementA::value);

    py::class_<ElementList, std::shared_ptr<ElementList>>(m2, "ElementList")
        .def(py::init<>())
        .def("add", &ElementList::add)
        .def("get", [](ElementList &el) {
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

    // Issue 389: r_v_p::move should fall-through to copy on non-movable objects
    class MoveIssue1 {
    public:
        MoveIssue1(int v) : v{v} {}
        MoveIssue1(const MoveIssue1 &c) { v = c.v; }
        MoveIssue1(MoveIssue1 &&) = delete;
        int v;
    };
    class MoveIssue2 {
    public:
        MoveIssue2(int v) : v{v} {}
        MoveIssue2(MoveIssue2 &&) = default;
        int v;
    };
    py::class_<MoveIssue1>(m2, "MoveIssue1").def(py::init<int>()).def_readwrite("value", &MoveIssue1::v);
    py::class_<MoveIssue2>(m2, "MoveIssue2").def(py::init<int>()).def_readwrite("value", &MoveIssue2::v);
    m2.def("get_moveissue1", [](int i) -> MoveIssue1 * { return new MoveIssue1(i); }, py::return_value_policy::move);
    m2.def("get_moveissue2", [](int i) { return MoveIssue2(i); }, py::return_value_policy::move);

    // Issues 392/397: overridding reference-returning functions
    class OverrideTest {
    public:
        struct A { std::string value = "hi"; };
        std::string v;
        A a;
        explicit OverrideTest(const std::string &v) : v{v} {}
        virtual std::string str_value() { return v; }
        virtual std::string &str_ref() { return v; }
        virtual A A_value() { return a; }
        virtual A &A_ref() { return a; }
    };
    class PyOverrideTest : public OverrideTest {
    public:
        using OverrideTest::OverrideTest;
        std::string str_value() override { PYBIND11_OVERLOAD(std::string, OverrideTest, str_value); }
        // Not allowed (uncommenting should hit a static_assert failure): we can't get a reference
        // to a python numeric value, since we only copy values in the numeric type caster:
//      std::string &str_ref() override { PYBIND11_OVERLOAD(std::string &, OverrideTest, str_ref); }
        // But we can work around it like this:
    private:
        std::string _tmp;
        std::string str_ref_helper() { PYBIND11_OVERLOAD(std::string, OverrideTest, str_ref); }
    public:
        std::string &str_ref() override { return _tmp = str_ref_helper(); }

        A A_value() override { PYBIND11_OVERLOAD(A, OverrideTest, A_value); }
        A &A_ref() override { PYBIND11_OVERLOAD(A &, OverrideTest, A_ref); }
    };
    py::class_<OverrideTest::A>(m2, "OverrideTest_A")
        .def_readwrite("value", &OverrideTest::A::value);
    py::class_<OverrideTest, PyOverrideTest>(m2, "OverrideTest")
        .def(py::init<const std::string &>())
        .def("str_value", &OverrideTest::str_value)
//      .def("str_ref", &OverrideTest::str_ref)
        .def("A_value", &OverrideTest::A_value)
        .def("A_ref", &OverrideTest::A_ref);

    /// Issue 393: need to return NotSupported to ensure correct arithmetic operator behavior
    py::class_<OpTest1>(m2, "OpTest1")
        .def(py::init<>())
        .def(py::self + py::self);

    py::class_<OpTest2>(m2, "OpTest2")
        .def(py::init<>())
        .def(py::self + py::self)
        .def("__add__", [](const OpTest2& c2, const OpTest1& c1) { return c2 + c1; })
        .def("__radd__", [](const OpTest2& c2, const OpTest1& c1) { return c2 + c1; });

    // Issue 388: Can't make iterators via make_iterator() with different r/v policies
    static std::vector<int> list = { 1, 2, 3 };
    m2.def("make_iterator_1", []() { return py::make_iterator<py::return_value_policy::copy>(list); });
    m2.def("make_iterator_2", []() { return py::make_iterator<py::return_value_policy::automatic>(list); });

    static std::vector<std::string> nothrows;
    // Issue 461: registering two things with the same name:
    py::class_<Dupe1>(m2, "Dupe1")
        .def("get_value", &Dupe1::get_value)
        ;
    m2.def("dupe1_factory", [](int v) { return new Dupe1(v); });

    py::class_<Dupe2>(m2, "Dupe2");
    py::exception<DupeException>(m2, "DupeException");

    try {
        m2.def("Dupe1", [](int v) { return new Dupe1(v); });
        nothrows.emplace_back("Dupe1");
    }
    catch (std::runtime_error &) {}
    try {
        py::class_<Dupe3>(m2, "dupe1_factory");
        nothrows.emplace_back("dupe1_factory");
    }
    catch (std::runtime_error &) {}
    try {
        py::exception<Dupe3>(m2, "Dupe2");
        nothrows.emplace_back("Dupe2");
    }
    catch (std::runtime_error &) {}
    try {
        m2.def("DupeException", []() { return 30; });
        nothrows.emplace_back("DupeException1");
    }
    catch (std::runtime_error &) {}
    try {
        py::class_<DupeException>(m2, "DupeException");
        nothrows.emplace_back("DupeException2");
    }
    catch (std::runtime_error &) {}
    m2.def("dupe_exception_failures", []() {
        py::list l;
        for (auto &e : nothrows) l.append(py::cast(e));
        return l;
    });

    /// Issue #471: shared pointer instance not dellocated
    class SharedChild : public std::enable_shared_from_this<SharedChild> {
    public:
        SharedChild() { print_created(this); }
        ~SharedChild() { print_destroyed(this); }
    };

    class SharedParent {
    public:
        SharedParent() : child(std::make_shared<SharedChild>()) { }
        const SharedChild &get_child() const { return *child; }

    private:
        std::shared_ptr<SharedChild> child;
    };

    py::class_<SharedChild, std::shared_ptr<SharedChild>>(m, "SharedChild");
    py::class_<SharedParent, std::shared_ptr<SharedParent>>(m, "SharedParent")
        .def(py::init<>())
        .def("get_child", &SharedParent::get_child, py::return_value_policy::reference);

    /// Issue/PR #478: unique ptrs constructed and freed without destruction
    class SpecialHolderObj {
    public:
        int val = 0;
        SpecialHolderObj *ch = nullptr;
        SpecialHolderObj(int v, bool make_child = true) : val{v}, ch{make_child ? new SpecialHolderObj(val+1, false) : nullptr}
        { print_created(this, val); }
        ~SpecialHolderObj() { delete ch; print_destroyed(this); }
        SpecialHolderObj *child() { return ch; }
    };

    py::class_<SpecialHolderObj, custom_unique_ptr<SpecialHolderObj>>(m, "SpecialHolderObj")
        .def(py::init<int>())
        .def("child", &SpecialHolderObj::child, pybind11::return_value_policy::reference_internal)
        .def_readwrite("val", &SpecialHolderObj::val)
        .def_static("holder_cstats", &ConstructorStats::get<custom_unique_ptr<SpecialHolderObj>>,
                py::return_value_policy::reference);

    /// Issue #484: number conversion generates unhandled exceptions
    m2.def("test_complex", [](float x) { py::print("{}"_s.format(x)); });
    m2.def("test_complex", [](std::complex<float> x) { py::print("({}, {})"_s.format(x.real(), x.imag())); });

    /// Issue #511: problem with inheritance + overwritten def_static
    struct MyBase {
        static std::unique_ptr<MyBase> make() {
            return std::unique_ptr<MyBase>(new MyBase());
        }
    };

    struct MyDerived : MyBase {
        static std::unique_ptr<MyDerived> make() {
            return std::unique_ptr<MyDerived>(new MyDerived());
        }
    };

    py::class_<MyBase>(m2, "MyBase")
        .def_static("make", &MyBase::make);

    py::class_<MyDerived, MyBase>(m2, "MyDerived")
        .def_static("make", &MyDerived::make)
        .def_static("make2", &MyDerived::make);

    py::dict d;
    std::string bar = "bar";
    d["str"] = bar;
    d["num"] = 3.7;

    /// Issue #528: templated constructor
    m2.def("tpl_constr_vector", [](std::vector<TplConstrClass> &) {});
    m2.def("tpl_constr_map", [](std::unordered_map<TplConstrClass, TplConstrClass> &) {});
    m2.def("tpl_constr_set", [](std::unordered_set<TplConstrClass> &) {});
#if defined(PYBIND11_HAS_OPTIONAL)
    m2.def("tpl_constr_optional", [](std::optional<TplConstrClass> &) {});
#elif defined(PYBIND11_HAS_EXP_OPTIONAL)
    m2.def("tpl_constr_optional", [](std::experimental::optional<TplConstrClass> &) {});
#endif
}

// MSVC workaround: trying to use a lambda here crashes MSCV
test_initializer issues(&init_issues);
