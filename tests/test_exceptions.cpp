/*
    tests/test_custom-exceptions.cpp -- exception translation

    Copyright (c) 2016 Pim Schellart <P.Schellart@princeton.edu>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/
#include "test_exceptions.h"

#include "local_bindings.h"
#include "pybind11_tests.h"

#include <exception>
#include <stdexcept>
#include <utility>

// A type that should be raised as an exception in Python
class MyException : public std::exception {
public:
    explicit MyException(const char *m) : message{m} {}
    const char *what() const noexcept override { return message.c_str(); }

private:
    std::string message = "";
};

// A type that should be translated to a standard Python exception
class MyException2 : public std::exception {
public:
    explicit MyException2(const char *m) : message{m} {}
    const char *what() const noexcept override { return message.c_str(); }

private:
    std::string message = "";
};

// A type that is not derived from std::exception (and is thus unknown)
class MyException3 {
public:
    explicit MyException3(const char *m) : message{m} {}
    virtual const char *what() const noexcept { return message.c_str(); }
    // Rule of 5 BEGIN: to preempt compiler warnings.
    MyException3(const MyException3 &) = default;
    MyException3(MyException3 &&) = default;
    MyException3 &operator=(const MyException3 &) = default;
    MyException3 &operator=(MyException3 &&) = default;
    virtual ~MyException3() = default;
    // Rule of 5 END.
private:
    std::string message = "";
};

// A type that should be translated to MyException
// and delegated to its exception translator
class MyException4 : public std::exception {
public:
    explicit MyException4(const char *m) : message{m} {}
    const char *what() const noexcept override { return message.c_str(); }

private:
    std::string message = "";
};

// Like the above, but declared via the helper function
class MyException5 : public std::logic_error {
public:
    explicit MyException5(const std::string &what) : std::logic_error(what) {}
};

// Inherits from MyException5
class MyException5_1 : public MyException5 {
    using MyException5::MyException5;
};

// Exception that will be caught via the module local translator.
class MyException6 : public std::exception {
public:
    explicit MyException6(const char *m) : message{m} {}
    const char *what() const noexcept override { return message.c_str(); }

private:
    std::string message = "";
};

struct PythonCallInDestructor {
    explicit PythonCallInDestructor(const py::dict &d) : d(d) {}
    ~PythonCallInDestructor() { d["good"] = true; }

    py::dict d;
};

struct PythonAlreadySetInDestructor {
    explicit PythonAlreadySetInDestructor(const py::str &s) : s(s) {}
    ~PythonAlreadySetInDestructor() {
        py::dict foo;
        try {
            // Assign to a py::object to force read access of nonexistent dict entry
            py::object o = foo["bar"];
        } catch (py::error_already_set &ex) {
            ex.discard_as_unraisable(s);
        }
    }

    py::str s;
};

TEST_SUBMODULE(exceptions, m) {
    m.def("throw_std_exception",
          []() { throw std::runtime_error("This exception was intentionally thrown."); });

    // make a new custom exception and use it as a translation target
    static py::exception<MyException> ex(m, "MyException");
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const MyException &e) {
            // Set MyException as the active python error
            ex(e.what());
        }
    });

    // register new translator for MyException2
    // no need to store anything here because this type will
    // never by visible from Python
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const MyException2 &e) {
            // Translate this exception to a standard RuntimeError
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });

    // register new translator for MyException4
    // which will catch it and delegate to the previously registered
    // translator for MyException by throwing a new exception
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const MyException4 &e) {
            throw MyException(e.what());
        }
    });

    // A simple exception translation:
    auto ex5 = py::register_exception<MyException5>(m, "MyException5");
    // A slightly more complicated one that declares MyException5_1 as a subclass of MyException5
    py::register_exception<MyException5_1>(m, "MyException5_1", ex5.ptr());

    // py::register_local_exception<LocalSimpleException>(m, "LocalSimpleException")

    py::register_local_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const MyException6 &e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });

    m.def("throws1", []() { throw MyException("this error should go to a custom type"); });
    m.def("throws2",
          []() { throw MyException2("this error should go to a standard Python exception"); });
    m.def("throws3", []() { throw MyException3("this error cannot be translated"); });
    m.def("throws4", []() { throw MyException4("this error is rethrown"); });
    m.def("throws5",
          []() { throw MyException5("this is a helper-defined translated exception"); });
    m.def("throws5_1", []() { throw MyException5_1("MyException5 subclass"); });
    m.def("throws6", []() { throw MyException6("MyException6 only handled in this module"); });
    m.def("throws_logic_error", []() {
        throw std::logic_error("this error should fall through to the standard handler");
    });
    m.def("throws_overflow_error", []() { throw std::overflow_error(""); });
    m.def("throws_local_error", []() { throw LocalException("never caught"); });
    m.def("throws_local_simple_error", []() { throw LocalSimpleException("this mod"); });
    m.def("exception_matches", []() {
        py::dict foo;
        try {
            // Assign to a py::object to force read access of nonexistent dict entry
            py::object o = foo["bar"];
        } catch (py::error_already_set &ex) {
            if (!ex.matches(PyExc_KeyError)) {
                throw;
            }
            return true;
        }
        return false;
    });
    m.def("exception_matches_base", []() {
        py::dict foo;
        try {
            // Assign to a py::object to force read access of nonexistent dict entry
            py::object o = foo["bar"];
        } catch (py::error_already_set &ex) {
            if (!ex.matches(PyExc_Exception)) {
                throw;
            }
            return true;
        }
        return false;
    });
    m.def("modulenotfound_exception_matches_base", []() {
        try {
            // On Python >= 3.6, this raises a ModuleNotFoundError, a subclass of ImportError
            py::module_::import("nonexistent");
        } catch (py::error_already_set &ex) {
            if (!ex.matches(PyExc_ImportError)) {
                throw;
            }
            return true;
        }
        return false;
    });

    m.def("throw_already_set", [](bool err) {
        if (err) {
            PyErr_SetString(PyExc_ValueError, "foo");
        }
        try {
            throw py::error_already_set();
        } catch (const std::runtime_error &e) {
            if ((err && e.what() != std::string("ValueError: foo"))
                || (!err && e.what() != std::string("Unknown internal error occurred"))) {
                PyErr_Clear();
                throw std::runtime_error("error message mismatch");
            }
        }
        PyErr_Clear();
        if (err) {
            PyErr_SetString(PyExc_ValueError, "foo");
        }
        throw py::error_already_set();
    });

    m.def("python_call_in_destructor", [](const py::dict &d) {
        bool retval = false;
        try {
            PythonCallInDestructor set_dict_in_destructor(d);
            PyErr_SetString(PyExc_ValueError, "foo");
            throw py::error_already_set();
        } catch (const py::error_already_set &) {
            retval = true;
        }
        return retval;
    });

    m.def("python_alreadyset_in_destructor", [](const py::str &s) {
        PythonAlreadySetInDestructor alreadyset_in_destructor(s);
        return true;
    });

    // test_nested_throws
    m.def("try_catch",
          [m](const py::object &exc_type, const py::function &f, const py::args &args) {
              try {
                  f(*args);
              } catch (py::error_already_set &ex) {
                  if (ex.matches(exc_type)) {
                      py::print(ex.what());
                  } else {
                      throw;
                  }
              }
          });

    // Test repr that cannot be displayed
    m.def("simple_bool_passthrough", [](bool x) { return x; });

    m.def("throw_should_be_translated_to_key_error", []() { throw shared_exception(); });

    m.def("raise_from", []() {
        PyErr_SetString(PyExc_ValueError, "inner");
        py::raise_from(PyExc_ValueError, "outer");
        throw py::error_already_set();
    });

    m.def("raise_from_already_set", []() {
        try {
            PyErr_SetString(PyExc_ValueError, "inner");
            throw py::error_already_set();
        } catch (py::error_already_set &e) {
            py::raise_from(e, PyExc_ValueError, "outer");
            throw py::error_already_set();
        }
    });

    m.def("throw_nested_exception", []() {
        try {
            throw std::runtime_error("Inner Exception");
        } catch (const std::runtime_error &) {
            std::throw_with_nested(std::runtime_error("Outer Exception"));
        }
    });
}
