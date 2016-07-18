/*
    example/example-custom-exceptions.cpp -- exception translation

    Copyright (c) 2016 Pim Schellart <P.Schellart@princeton.edu>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"

// A type that should be raised as an exeption in Python
class MyException : public std::exception {
public:
    explicit MyException(const char * m) : message{m} {}
    virtual const char * what() const noexcept override {return message.c_str();}
private:
    std::string message = "";
};

// A type that should be translated to a standard Python exception
class MyException2 : public std::exception {
public:
    explicit MyException2(const char * m) : message{m} {}
    virtual const char * what() const noexcept override {return message.c_str();}
private:
    std::string message = "";
};

// A type that is not derived from std::exception (and is thus unknown)
class MyException3 {
public:
    explicit MyException3(const char * m) : message{m} {}
    virtual const char * what() const noexcept {return message.c_str();}
private:
    std::string message = "";
};

// A type that should be translated to MyException
// and delegated to its exception translator
class MyException4 : public std::exception {
public:
    explicit MyException4(const char * m) : message{m} {}
    virtual const char * what() const noexcept override {return message.c_str();}
private:
    std::string message = "";
};

void throws1() {
    throw MyException("this error should go to a custom type");
}

void throws2() {
    throw MyException2("this error should go to a standard Python exception");
}

void throws3() {
    throw MyException3("this error cannot be translated");
}

void throws4() {
    throw MyException4("this error is rethrown");
}

void throws_logic_error() {
    throw std::logic_error("this error should fall through to the standard handler");
}

void init_ex_custom_exceptions(py::module &m) {
    // make a new custom exception and use it as a translation target
    static py::exception<MyException> ex(m, "MyException");
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const MyException &e) {
            PyErr_SetString(ex.ptr(), e.what());
        }
    });

    // register new translator for MyException2
    // no need to store anything here because this type will
    // never by visible from Python
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const MyException2 &e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });

    // register new translator for MyException4
    // which will catch it and delegate to the previously registered
    // translator for MyException by throwing a new exception
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const MyException4 &e) {
            throw MyException(e.what());
        }
    });

    m.def("throws1", &throws1);
    m.def("throws2", &throws2);
    m.def("throws3", &throws3);
    m.def("throws4", &throws4);
    m.def("throws_logic_error", &throws_logic_error);
}

