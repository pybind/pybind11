/*
    example/example3.cpp -- operator overloading

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"
#include <pybind11/operators.h>

class Vector2 {
public:
    Vector2(float x, float y) : x(x), y(y) { std::cout << "Value constructor" << std::endl; }
    Vector2(const Vector2 &v) : x(v.x), y(v.y) { std::cout << "Copy constructor" << std::endl; }
    Vector2(Vector2 &&v) : x(v.x), y(v.y) { std::cout << "Move constructor" << std::endl; v.x = v.y = 0; }
    ~Vector2() { std::cout << "Destructor." << std::endl; }

    std::string toString() const {
        return "[" + std::to_string(x) + ", " + std::to_string(y) + "]";
    }

    void operator=(const Vector2 &v) {
        cout << "Assignment operator" << endl;
        x = v.x;
        y = v.y;
    }

    void operator=(Vector2 &&v) {
        cout << "Move assignment operator" << endl;
        x = v.x; y = v.y; v.x = v.y = 0;
    }

    Vector2 operator+(const Vector2 &v) const { return Vector2(x + v.x, y + v.y); }
    Vector2 operator-(const Vector2 &v) const { return Vector2(x - v.x, y - v.y); }
    Vector2 operator-(float value) const { return Vector2(x - value, y - value); }
    Vector2 operator+(float value) const { return Vector2(x + value, y + value); }
    Vector2 operator*(float value) const { return Vector2(x * value, y * value); }
    Vector2 operator/(float value) const { return Vector2(x / value, y / value); }
    Vector2& operator+=(const Vector2 &v) { x += v.x; y += v.y; return *this; }
    Vector2& operator-=(const Vector2 &v) { x -= v.x; y -= v.y; return *this; }
    Vector2& operator*=(float v) { x *= v; y *= v; return *this; }
    Vector2& operator/=(float v) { x /= v; y /= v; return *this; }

    friend Vector2 operator+(float f, const Vector2 &v) { return Vector2(f + v.x, f + v.y); }
    friend Vector2 operator-(float f, const Vector2 &v) { return Vector2(f - v.x, f - v.y); }
    friend Vector2 operator*(float f, const Vector2 &v) { return Vector2(f * v.x, f * v.y); }
    friend Vector2 operator/(float f, const Vector2 &v) { return Vector2(f / v.x, f / v.y); }
private:
    float x, y;
};


void init_ex3(py::module &m) {
    py::class_<Vector2>(m, "Vector2")
        .def(py::init<float, float>())
        .def(py::self + py::self)
        .def(py::self + float())
        .def(py::self - py::self)
        .def(py::self - float())
        .def(py::self * float())
        .def(py::self / float())
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def(py::self *= float())
        .def(py::self /= float())
        .def(float() + py::self)
        .def(float() - py::self)
        .def(float() * py::self)
        .def(float() / py::self)
        .def("__str__", &Vector2::toString);

    m.attr("Vector") = m.attr("Vector2");
}
