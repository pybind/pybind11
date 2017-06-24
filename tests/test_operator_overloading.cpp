/*
    tests/test_operator_overloading.cpp -- operator overloading

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include "constructor_stats.h"
#include <pybind11/operators.h>

class Vector2 {
public:
    Vector2(float x, float y) : x(x), y(y) { print_created(this, toString()); }
    Vector2(const Vector2 &v) : x(v.x), y(v.y) { print_copy_created(this); }
    Vector2(Vector2 &&v) : x(v.x), y(v.y) { print_move_created(this); v.x = v.y = 0; }
    ~Vector2() { print_destroyed(this); }

    std::string toString() const {
        return "[" + std::to_string(x) + ", " + std::to_string(y) + "]";
    }

    void operator=(const Vector2 &v) {
        print_copy_assigned(this);
        x = v.x;
        y = v.y;
    }

    void operator=(Vector2 &&v) {
        print_move_assigned(this);
        x = v.x; y = v.y; v.x = v.y = 0;
    }

    Vector2 operator+(const Vector2 &v) const { return Vector2(x + v.x, y + v.y); }
    Vector2 operator-(const Vector2 &v) const { return Vector2(x - v.x, y - v.y); }
    Vector2 operator-(float value) const { return Vector2(x - value, y - value); }
    Vector2 operator+(float value) const { return Vector2(x + value, y + value); }
    Vector2 operator*(float value) const { return Vector2(x * value, y * value); }
    Vector2 operator/(float value) const { return Vector2(x / value, y / value); }
    Vector2 operator*(const Vector2 &v) const { return Vector2(x * v.x, y * v.y); }
    Vector2 operator/(const Vector2 &v) const { return Vector2(x / v.x, y / v.y); }
    Vector2& operator+=(const Vector2 &v) { x += v.x; y += v.y; return *this; }
    Vector2& operator-=(const Vector2 &v) { x -= v.x; y -= v.y; return *this; }
    Vector2& operator*=(float v) { x *= v; y *= v; return *this; }
    Vector2& operator/=(float v) { x /= v; y /= v; return *this; }
    Vector2& operator*=(const Vector2 &v) { x *= v.x; y *= v.y; return *this; }
    Vector2& operator/=(const Vector2 &v) { x /= v.x; y /= v.y; return *this; }

    friend Vector2 operator+(float f, const Vector2 &v) { return Vector2(f + v.x, f + v.y); }
    friend Vector2 operator-(float f, const Vector2 &v) { return Vector2(f - v.x, f - v.y); }
    friend Vector2 operator*(float f, const Vector2 &v) { return Vector2(f * v.x, f * v.y); }
    friend Vector2 operator/(float f, const Vector2 &v) { return Vector2(f / v.x, f / v.y); }
private:
    float x, y;
};

class C1 { };
class C2 { };

int operator+(const C1 &, const C1 &) { return 11; }
int operator+(const C2 &, const C2 &) { return 22; }
int operator+(const C2 &, const C1 &) { return 21; }
int operator+(const C1 &, const C2 &) { return 12; }

struct NestABase {
    int value = -2;
};

struct NestA : NestABase {
    int value = 3;
    NestA& operator+=(int i) { value += i; return *this; }
};

struct NestB {
    NestA a;
    int value = 4;
    NestB& operator-=(int i) { value -= i; return *this; }
};

struct NestC {
    NestB b;
    int value = 5;
    NestC& operator*=(int i) { value *= i; return *this; }
};

test_initializer operator_overloading([](py::module &pm) {
    auto m = pm.def_submodule("operators");

    py::class_<Vector2>(m, "Vector2")
        .def(py::init<float, float>())
        .def(py::self + py::self)
        .def(py::self + float())
        .def(py::self - py::self)
        .def(py::self - float())
        .def(py::self * float())
        .def(py::self / float())
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def(py::self *= float())
        .def(py::self /= float())
        .def(py::self *= py::self)
        .def(py::self /= py::self)
        .def(float() + py::self)
        .def(float() - py::self)
        .def(float() * py::self)
        .def(float() / py::self)
        .def("__str__", &Vector2::toString)
        ;

    m.attr("Vector") = m.attr("Vector2");

    // #393: need to return NotSupported to ensure correct arithmetic operator behavior
    py::class_<C1>(m, "C1")
        .def(py::init<>())
        .def(py::self + py::self);

    py::class_<C2>(m, "C2")
        .def(py::init<>())
        .def(py::self + py::self)
        .def("__add__", [](const C2& c2, const C1& c1) { return c2 + c1; })
        .def("__radd__", [](const C2& c2, const C1& c1) { return c1 + c2; });

    // #328: first member in a class can't be used in operators
    py::class_<NestABase>(m, "NestABase")
        .def(py::init<>())
        .def_readwrite("value", &NestABase::value);

    py::class_<NestA>(m, "NestA")
        .def(py::init<>())
        .def(py::self += int())
        .def("as_base", [](NestA &a) -> NestABase& {
            return (NestABase&) a;
        }, py::return_value_policy::reference_internal);

    py::class_<NestB>(m, "NestB")
        .def(py::init<>())
        .def(py::self -= int())
        .def_readwrite("a", &NestB::a);

    py::class_<NestC>(m, "NestC")
        .def(py::init<>())
        .def(py::self *= int())
        .def_readwrite("b", &NestC::b);

    m.def("get_NestA", [](const NestA &a) { return a.value; });
    m.def("get_NestB", [](const NestB &b) { return b.value; });
    m.def("get_NestC", [](const NestC &c) { return c.value; });
});
