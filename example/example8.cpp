/*
    example/example8.cpp -- binding classes with custom reference counting,
    implicit conversions between types

    Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"
#include "object.h"

/// Object subclass
class MyObject : public Object {
public:
    MyObject(int value) : value(value) {
        std::cout << toString() << " constructor" << std::endl;
    }

    std::string toString() const {
        return "MyObject[" + std::to_string(value) + "]";
    }

protected:
    virtual ~MyObject() {
        std::cout << toString() << " destructor" << std::endl;
    }

private:
    int value;
};

/// Make pybind aware of the ref-counted wrapper type
namespace pybind { namespace detail {
template <typename T> class type_caster<ref<T>>
    : public type_caster_holder<T, ref<T>> { };
}}

Object *make_object_1() { return new MyObject(1); }
ref<Object> make_object_2() { return new MyObject(2); }
MyObject *make_myobject_4() { return new MyObject(4); }
ref<MyObject> make_myobject_5() { return new MyObject(5); }

void print_object_1(const Object *obj) { std::cout << obj->toString() << std::endl; }
void print_object_2(ref<Object> obj) { std::cout << obj->toString() << std::endl; }
void print_object_3(const ref<Object> &obj) { std::cout << obj->toString() << std::endl; }
void print_object_4(const ref<Object> *obj) { std::cout << (*obj)->toString() << std::endl; }

void print_myobject_1(const MyObject *obj) { std::cout << obj->toString() << std::endl; }
void print_myobject_2(ref<MyObject> obj) { std::cout << obj->toString() << std::endl; }
void print_myobject_3(const ref<MyObject> &obj) { std::cout << obj->toString() << std::endl; }
void print_myobject_4(const ref<MyObject> *obj) { std::cout << (*obj)->toString() << std::endl; }

void init_ex8(py::module &m) {
    py::class_<Object, ref<Object>> obj(m, "Object");
    obj.def("getRefCount", &Object::getRefCount);

    py::class_<MyObject, ref<MyObject>>(m, "MyObject", obj)
        .def(py::init<int>());

    m.def("make_object_1", &make_object_1);
    m.def("make_object_2", &make_object_2);
    m.def("make_myobject_4", &make_myobject_4);
    m.def("make_myobject_5", &make_myobject_5);
    m.def("print_object_1", &print_object_1);
    m.def("print_object_2", &print_object_2);
    m.def("print_object_3", &print_object_3);
    m.def("print_object_4", &print_object_4);
    m.def("print_myobject_1", &print_myobject_1);
    m.def("print_myobject_2", &print_myobject_2);
    m.def("print_myobject_3", &print_myobject_3);
    m.def("print_myobject_4", &print_myobject_4);

    py::implicitly_convertible<py::int_, MyObject>();
}
