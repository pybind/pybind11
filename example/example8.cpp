/*
    example/example8.cpp -- binding classes with custom reference counting,
    implicit conversions between types

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"
#include "object.h"

/// Custom object with builtin reference counting (see 'object.h' for the implementation)
class MyObject1 : public Object {
public:
    MyObject1(int value) : value(value) {
        std::cout << toString() << " constructor" << std::endl;
    }

    std::string toString() const {
        return "MyObject1[" + std::to_string(value) + "]";
    }

protected:
    virtual ~MyObject1() {
        std::cout << toString() << " destructor" << std::endl;
    }

private:
    int value;
};

/// Object managed by a std::shared_ptr<>
class MyObject2 {
public:
    MyObject2(int value) : value(value) {
        std::cout << toString() << " constructor" << std::endl;
    }

    std::string toString() const {
        return "MyObject2[" + std::to_string(value) + "]";
    }

    virtual ~MyObject2() {
        std::cout << toString() << " destructor" << std::endl;
    }

private:
    int value;
};

/// Object managed by a std::shared_ptr<>, additionally derives from std::enable_shared_from_this<>
class MyObject3 : public std::enable_shared_from_this<MyObject3> {
public:
    MyObject3(int value) : value(value) {
        std::cout << toString() << " constructor" << std::endl;
    }

    std::string toString() const {
        return "MyObject3[" + std::to_string(value) + "]";
    }

    virtual ~MyObject3() {
        std::cout << toString() << " destructor" << std::endl;
    }

private:
    int value;
};

/// Make pybind aware of the ref-counted wrapper type (s)
PYBIND11_DECLARE_HOLDER_TYPE(T, ref<T>);
PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

Object *make_object_1() { return new MyObject1(1); }
ref<Object> make_object_2() { return new MyObject1(2); }

MyObject1 *make_myobject1_1() { return new MyObject1(4); }
ref<MyObject1> make_myobject1_2() { return new MyObject1(5); }

MyObject2 *make_myobject2_1() { return new MyObject2(6); }
std::shared_ptr<MyObject2> make_myobject2_2() { return std::make_shared<MyObject2>(7); }

MyObject3 *make_myobject3_1() { return new MyObject3(8); }
std::shared_ptr<MyObject3> make_myobject3_2() { return std::make_shared<MyObject3>(9); }

void print_object_1(const Object *obj) { std::cout << obj->toString() << std::endl; }
void print_object_2(ref<Object> obj) { std::cout << obj->toString() << std::endl; }
void print_object_3(const ref<Object> &obj) { std::cout << obj->toString() << std::endl; }
void print_object_4(const ref<Object> *obj) { std::cout << (*obj)->toString() << std::endl; }

void print_myobject1_1(const MyObject1 *obj) { std::cout << obj->toString() << std::endl; }
void print_myobject1_2(ref<MyObject1> obj) { std::cout << obj->toString() << std::endl; }
void print_myobject1_3(const ref<MyObject1> &obj) { std::cout << obj->toString() << std::endl; }
void print_myobject1_4(const ref<MyObject1> *obj) { std::cout << (*obj)->toString() << std::endl; }

void print_myobject2_1(const MyObject2 *obj) { std::cout << obj->toString() << std::endl; }
void print_myobject2_2(std::shared_ptr<MyObject2> obj) { std::cout << obj->toString() << std::endl; }
void print_myobject2_3(const std::shared_ptr<MyObject2> &obj) { std::cout << obj->toString() << std::endl; }
void print_myobject2_4(const std::shared_ptr<MyObject2> *obj) { std::cout << (*obj)->toString() << std::endl; }

void print_myobject3_1(const MyObject3 *obj) { std::cout << obj->toString() << std::endl; }
void print_myobject3_2(std::shared_ptr<MyObject3> obj) { std::cout << obj->toString() << std::endl; }
void print_myobject3_3(const std::shared_ptr<MyObject3> &obj) { std::cout << obj->toString() << std::endl; }
void print_myobject3_4(const std::shared_ptr<MyObject3> *obj) { std::cout << (*obj)->toString() << std::endl; }

void init_ex8(py::module &m) {
    py::class_<Object, ref<Object>> obj(m, "Object");
    obj.def("getRefCount", &Object::getRefCount);

    py::class_<MyObject1, ref<MyObject1>>(m, "MyObject1", obj)
        .def(py::init<int>());

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

    py::implicitly_convertible<py::int_, MyObject1>();
}
