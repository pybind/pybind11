/*
    tests/test_implicit_conversion.cpp -- implicit conversion between types

    Copyright (c) 2016 Jason Rhinelander <jason@imaginary.ca>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include "constructor_stats.h"
#include <cmath>

/// Objects to test implicit conversion
class ExIC_A {
public:
    // Implicit conversion *from* double
    ExIC_A(double v) : value{v} { print_created(this, v); }
    // Default constructor
    ExIC_A() : value{42.0} { print_default_created(this); }
    // Implicit conversion *to* double
    virtual operator double() const { print_values(this, "double conversion operator"); return value; }
    virtual ~ExIC_A() { print_destroyed(this); }
private:
    double value;
};
class ExIC_E;
class ExIC_B : public ExIC_A {
public:
    ExIC_B() { print_default_created(this); }
    ~ExIC_B() { print_destroyed(this); }
    ExIC_B(const ExIC_B &b) : ExIC_A(b) { print_copy_created(this); }
    // Implicit conversion to ExIC_E
    operator ExIC_E() const;
};
class ExIC_C : public ExIC_B {
public:
    ExIC_C() { print_default_created(this); }
    ~ExIC_C() { print_destroyed(this); }
    ExIC_C(const ExIC_C &c) : ExIC_B(c) { print_copy_created(this); }
    // Implicit conversion to double
    virtual operator double() const override { return 3.14159265358979323846; }
    // Implicit conversion to string
    operator std::string() const { return "pi"; }
};
class ExIC_D : public ExIC_A {
public:
    ExIC_D() { print_default_created(this); }
    ~ExIC_D() { print_destroyed(this); }
    ExIC_D(const ExIC_D &d) : ExIC_A(d) { print_copy_created(this); }
    // Implicit conversion to double
    virtual operator double() const override { return 2.71828182845904523536; }
    // Implicit conversion to string
    operator std::string() const { return "e"; }
};
// This class won't be registered with pybind11, but a function accepting it will be--the function
// can only be called with arguments that are implicitly convertible to ExIC_E
class ExIC_E {
public:
    ExIC_E() = delete;
    ExIC_E(const ExIC_E &e) : value{e.value} { print_copy_created(this); }
    ExIC_E(ExIC_E &&e) : value{std::move(e.value)} { print_move_created(this); }
    ~ExIC_E() { print_destroyed(this); }
    // explicit constructors should not be called by implicit conversion:
    explicit ExIC_E(double d) : value{d} { print_created(this, "double constructor", d); }
    explicit ExIC_E(const ExIC_A &a) : value{(double)a / 3.0} { print_created(this, "ExIC_A conversion constructor"); }
    // Convertible implicitly from D:
    ExIC_E(const ExIC_D &d) : value{3*d} { print_created(this, "ExIC_D conversion constructor"); }
    // Implicit conversion to double:
    operator double() const { print_values(this, "double conversion operator"); return value; }
private:
    double value;
};
ExIC_B::operator ExIC_E() const { print_values(this, "ExIC_E conversion operator");  return ExIC_E(2*(double)(*this)); }
// Class without a move constructor (just to be sure we don't depend on a move constructor).
// Unlike the above, we *will* expose this one to python, but will declare its
// implicitly_convertible before registering it, which will result in C++ (not python) type
// conversion.
class ExIC_F {
public:
    ExIC_F() : value{99.0} { print_default_created(this); }
    ExIC_F(const ExIC_A &a) : value{(double)a*1000} { print_created(this, " xIC_A conversion constructor"); }
    ExIC_F(const ExIC_F &f) : value{f.value} { print_copy_created(this); }
    ~ExIC_F() { print_destroyed(this); }
    operator double() const { print_values(this, "double conversion operator"); return value; }
private:
    double value;
};

class ExIC_G1 {
public:
    operator long() const { return 111; }
};
class ExIC_G2 : public ExIC_G1 {
public:
    operator long() const { return 222; }
};
class ExIC_G3 {
public:
    operator long() const { return 333; }
};
class ExIC_G4 : public ExIC_G3 {
public:
    operator long() const { return 444; }
};

// Implicit conversion methods; the ones that just return themselves are called with an object, i.e.
// implicit conversion of the function arguments to fit the function.
double as_double(double d) { return d; }
long as_long(long l) { return l; }
std::string as_string(const std::string &s) { return s; }
double double_exICe(const ExIC_E &e) { return (double) e; }
double double_exICf(const ExIC_F &f) { return (double) f; }

void init_ex_implicit_conversion(py::module &m) {

    py::class_<ExIC_A> a(m, "ExIC_A");
    a.def(py::init<>());
    a.def(py::init<double>());

    // We can construct a ExIC_A from a double:
    py::implicitly_convertible<py::float_, ExIC_A>();

    // It can also be implicitly to a double:
    py::implicitly_convertible<ExIC_A, double>();


    py::class_<ExIC_B> b(m, "ExIC_B", a);
    b.def(py::init<>());

    py::class_<ExIC_C> c(m, "ExIC_C", b);
    c.def(py::init<>());

    py::class_<ExIC_D> d(m, "ExIC_D", a);
    d.def(py::init<>());

    // NB: don't need to implicitly declare ExIC_{B,C} as convertible to double: they automatically
    // get that since we told pybind11 they inherit from A
    py::implicitly_convertible<ExIC_C, std::string>();
    py::implicitly_convertible<ExIC_D, std::string>();

    // NB: ExIC_E is a non-pybind-registered class:
    //
    // This should fail: ExIC_A is *not* C++ implicitly convertible to ExIC_E (the constructor is
    // marked explicit):
    try {
        py::implicitly_convertible<ExIC_A, ExIC_E>();
        std::cout << "py::implicitly_convertible<ExIC_A, ExIC_E>() should have thrown, but didn't!" << std::endl;
    }
    catch (std::runtime_error) {}

    py::implicitly_convertible<ExIC_B, ExIC_E>();
    // This isn't needed, since pybind knows C inherits from B
    //py::implicitly_convertible<ExIC_C, ExIC_E>();
    py::implicitly_convertible<ExIC_D, ExIC_E>();

    m.def("as_double", &as_double);
    m.def("as_long", &as_long);
    m.def("as_string", &as_string);
    m.def("double_exICe", &double_exICe);
    m.def("double_exICf", &double_exICf);

    // Here's how we can get C++-level implicit conversion even with a pybind-registered type: tell
    // pybind11 that the type is convertible to F before registering F:
    py::implicitly_convertible<ExIC_A, ExIC_F>();

    py::class_<ExIC_F> f(m, "ExIC_F");
    // We allow ExIC_F to be constructed in Python, but don't provide a conversion constructor from
    // ExIC_A.  C++ has an implicit one, however, that we told pybind11 about above.  In practice
    // this means we are allowed to pass ExIC_A instances to functions taking ExIC_F arguments, but
    // aren't allowed to write `exIC_func(ExIC_F(a))` because the explicit conversion is
    // (intentionally) not exposed to python.  (Whether this is useful is really up to the
    // developer).
    f.def(py::init<>());

    py::class_<ExIC_G1> g1(m, "ExIC_G1");     g1.def(py::init<>());
    py::class_<ExIC_G2> g2(m, "ExIC_G2", g1); g2.def(py::init<>());
    py::class_<ExIC_G3> g3(m, "ExIC_G3");     g3.def(py::init<>());
    py::class_<ExIC_G4> g4(m, "ExIC_G4", g3); g4.def(py::init<>());
    // Make sure that the order we declare convertibility doesn't matter: i.e. the base class
    // conversions here (G1 and G3) should not be invoked for G2 and G4, regardless of the
    // implicitly convertible declaration order.
    py::implicitly_convertible<ExIC_G2, long>();
    py::implicitly_convertible<ExIC_G1, long>();
    py::implicitly_convertible<ExIC_G3, long>();
    py::implicitly_convertible<ExIC_G4, long>();

    m.def("cstats_ExIC_E", &ConstructorStats::get<ExIC_E>);
}
