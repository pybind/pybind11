/*
    tests/test_smart_ptr.cpp -- binding classes with custom reference counting,
    implicit conversions between types

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include "object.h"

/// Custom object with builtin reference counting (see 'object.h' for the implementation)
class MyObject1 : public Object {
public:
    MyObject1(int value) : value(value) {
        print_created(this, toString());
    }

    std::string toString() const {
        return "MyObject1[" + std::to_string(value) + "]";
    }

protected:
    virtual ~MyObject1() {
        print_destroyed(this);
    }

private:
    int value;
};

/// Object managed by a std::shared_ptr<>
class MyObject2 {
public:
    MyObject2(int value) : value(value) {
        print_created(this, toString());
    }

    std::string toString() const {
        return "MyObject2[" + std::to_string(value) + "]";
    }

    virtual ~MyObject2() {
        print_destroyed(this);
    }

private:
    int value;
};

/// Object managed by a std::shared_ptr<>, additionally derives from std::enable_shared_from_this<>
class MyObject3 : public std::enable_shared_from_this<MyObject3> {
public:
    MyObject3(int value) : value(value) {
        print_created(this, toString());
    }

    std::string toString() const {
        return "MyObject3[" + std::to_string(value) + "]";
    }

    virtual ~MyObject3() {
        print_destroyed(this);
    }

private:
    int value;
};

class MyObject4 {
public:
    MyObject4(int value) : value{value} {
        print_created(this);
    }
    int value;
private:
    ~MyObject4() {
        print_destroyed(this);
    }
};

/// Make pybind aware of the ref-counted wrapper type (s)

// ref<T> is a wrapper for 'Object' which uses intrusive reference counting
// It is always possible to construct a ref<T> from an Object* pointer without
// possible incosistencies, hence the 'true' argument at the end.
PYBIND11_DECLARE_HOLDER_TYPE(T, ref<T>, true);
PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>); // Not required any more for std::shared_ptr,
                                                     // but it should compile without error

Object *make_object_1() { return new MyObject1(1); }
ref<Object> make_object_2() { return new MyObject1(2); }

MyObject1 *make_myobject1_1() { return new MyObject1(4); }
ref<MyObject1> make_myobject1_2() { return new MyObject1(5); }

MyObject2 *make_myobject2_1() { return new MyObject2(6); }
std::shared_ptr<MyObject2> make_myobject2_2() { return std::make_shared<MyObject2>(7); }

MyObject3 *make_myobject3_1() { return new MyObject3(8); }
std::shared_ptr<MyObject3> make_myobject3_2() { return std::make_shared<MyObject3>(9); }

void print_object_1(const Object *obj) { py::print(obj->toString()); }
void print_object_2(ref<Object> obj) { py::print(obj->toString()); }
void print_object_3(const ref<Object> &obj) { py::print(obj->toString()); }
void print_object_4(const ref<Object> *obj) { py::print((*obj)->toString()); }

void print_myobject1_1(const MyObject1 *obj) { py::print(obj->toString()); }
void print_myobject1_2(ref<MyObject1> obj) { py::print(obj->toString()); }
void print_myobject1_3(const ref<MyObject1> &obj) { py::print(obj->toString()); }
void print_myobject1_4(const ref<MyObject1> *obj) { py::print((*obj)->toString()); }

void print_myobject2_1(const MyObject2 *obj) { py::print(obj->toString()); }
void print_myobject2_2(std::shared_ptr<MyObject2> obj) { py::print(obj->toString()); }
void print_myobject2_3(const std::shared_ptr<MyObject2> &obj) { py::print(obj->toString()); }
void print_myobject2_4(const std::shared_ptr<MyObject2> *obj) { py::print((*obj)->toString()); }

void print_myobject3_1(const MyObject3 *obj) { py::print(obj->toString()); }
void print_myobject3_2(std::shared_ptr<MyObject3> obj) { py::print(obj->toString()); }
void print_myobject3_3(const std::shared_ptr<MyObject3> &obj) { py::print(obj->toString()); }
void print_myobject3_4(const std::shared_ptr<MyObject3> *obj) { py::print((*obj)->toString()); }

test_initializer smart_ptr([](py::module &m) {
    py::class_<Object, ref<Object>> obj(m, "Object");
    obj.def("getRefCount", &Object::getRefCount);

    py::class_<MyObject1, ref<MyObject1>>(m, "MyObject1", obj)
        .def(py::init<int>());

    m.def("test_object1_refcounting",
        []() -> bool {
            ref<MyObject1> o = new MyObject1(0);
            bool good = o->getRefCount() == 1;
            py::object o2 = py::cast(o, py::return_value_policy::reference);
            // always request (partial) ownership for objects with intrusive
            // reference counting even when using the 'reference' RVP
            good &= o->getRefCount() == 2;
            return good;
        }
    );

    m.def("make_object_1", &make_object_1);
    m.def("make_object_2", &make_object_2);
    m.def("make_myobject1_1", &make_myobject1_1);
    m.def("make_myobject1_2", &make_myobject1_2);
    m.def("print_object_1", &print_object_1);
    m.def("print_object_2", &print_object_2);
    m.def("print_object_3", &print_object_3);
    m.def("print_object_4", &print_object_4);
    m.def("print_myobject1_1", &print_myobject1_1);
    m.def("print_myobject1_2", &print_myobject1_2);
    m.def("print_myobject1_3", &print_myobject1_3);
    m.def("print_myobject1_4", &print_myobject1_4);

    py::class_<MyObject2, std::shared_ptr<MyObject2>>(m, "MyObject2")
        .def(py::init<int>());
    m.def("make_myobject2_1", &make_myobject2_1);
    m.def("make_myobject2_2", &make_myobject2_2);
    m.def("print_myobject2_1", &print_myobject2_1);
    m.def("print_myobject2_2", &print_myobject2_2);
    m.def("print_myobject2_3", &print_myobject2_3);
    m.def("print_myobject2_4", &print_myobject2_4);

    py::class_<MyObject3, std::shared_ptr<MyObject3>>(m, "MyObject3")
        .def(py::init<int>());
    m.def("make_myobject3_1", &make_myobject3_1);
    m.def("make_myobject3_2", &make_myobject3_2);
    m.def("print_myobject3_1", &print_myobject3_1);
    m.def("print_myobject3_2", &print_myobject3_2);
    m.def("print_myobject3_3", &print_myobject3_3);
    m.def("print_myobject3_4", &print_myobject3_4);

    py::class_<MyObject4, std::unique_ptr<MyObject4, py::nodelete>>(m, "MyObject4")
        .def(py::init<int>())
        .def_readwrite("value", &MyObject4::value);

    py::implicitly_convertible<py::int_, MyObject1>();

    // Expose constructor stats for the ref type
    m.def("cstats_ref", &ConstructorStats::get<ref_tag>);
});

struct SharedPtrRef {
    struct A {
        A() { print_created(this); }
        A(const A &) { print_copy_created(this); }
        A(A &&) { print_move_created(this); }
        ~A() { print_destroyed(this); }
    };

    A value = {};
    std::shared_ptr<A> shared = std::make_shared<A>();
};

struct SharedFromThisRef {
    struct B : std::enable_shared_from_this<B> {
        B() { print_created(this); }
        B(const B &) : std::enable_shared_from_this<B>() { print_copy_created(this); }
        B(B &&) : std::enable_shared_from_this<B>() { print_move_created(this); }
        ~B() { print_destroyed(this); }
    };

    B value = {};
    std::shared_ptr<B> shared = std::make_shared<B>();
};

test_initializer smart_ptr_and_references([](py::module &pm) {
    auto m = pm.def_submodule("smart_ptr");

    using A = SharedPtrRef::A;
    py::class_<A, std::shared_ptr<A>>(m, "A");

    py::class_<SharedPtrRef>(m, "SharedPtrRef")
        .def(py::init<>())
        .def_readonly("ref", &SharedPtrRef::value)
        .def_property_readonly("copy", [](const SharedPtrRef &s) { return s.value; },
                               py::return_value_policy::copy)
        .def_readonly("holder_ref", &SharedPtrRef::shared)
        .def_property_readonly("holder_copy", [](const SharedPtrRef &s) { return s.shared; },
                               py::return_value_policy::copy)
        .def("set_ref", [](SharedPtrRef &, const A &) { return true; })
        .def("set_holder", [](SharedPtrRef &, std::shared_ptr<A>) { return true; });

    using B = SharedFromThisRef::B;
    py::class_<B, std::shared_ptr<B>>(m, "B");

    py::class_<SharedFromThisRef>(m, "SharedFromThisRef")
        .def(py::init<>())
        .def_readonly("bad_wp", &SharedFromThisRef::value)
        .def_property_readonly("ref", [](const SharedFromThisRef &s) -> const B & { return *s.shared; })
        .def_property_readonly("copy", [](const SharedFromThisRef &s) { return s.value; },
                               py::return_value_policy::copy)
        .def_readonly("holder_ref", &SharedFromThisRef::shared)
        .def_property_readonly("holder_copy", [](const SharedFromThisRef &s) { return s.shared; },
                               py::return_value_policy::copy)
        .def("set_ref", [](SharedFromThisRef &, const B &) { return true; })
        .def("set_holder", [](SharedFromThisRef &, std::shared_ptr<B>) { return true; });
});
