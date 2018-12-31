/*
    tests/test_with.cpp -- Usage of pybind11 with statement

    Copyright (c) 2018 Yannick Jadoul

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

#include <pybind11/eval.h>

class CppContextManager {
public:
    CppContextManager(std::string value, bool swallow_exceptions=false)
        : m_value(std::move(value)), m_swallow_exceptions(swallow_exceptions) {}

    std::string enter() {
        ++m_entered;
        return m_value;
    }

    bool exit(py::args args) {
        ++m_exited;
        m_exit_args = args;
        return m_swallow_exceptions;
    }

    std::string m_value;
    bool m_swallow_exceptions;
    unsigned int m_entered = 0;
    unsigned int m_exited = 0;
    py::object m_exit_args = py::none();
};

class NewCppException {};


TEST_SUBMODULE(with_, m) {
    py::class_<CppContextManager>(m, "CppContextManager")
        .def(py::init<std::string, bool>(), py::arg("value"), py::arg("swallow_exceptions") = false)
        .def("__enter__", &CppContextManager::enter)
        .def("__exit__", &CppContextManager::exit)
        .def_readonly("value", &CppContextManager::m_value)
        .def_readonly("entered", &CppContextManager::m_entered)
        .def_readonly("exited", &CppContextManager::m_exited)
        .def_readonly("exit_args", &CppContextManager::m_exit_args);

    py::enum_<py::with_exception_policy>(m, "WithExceptionPolicy")
        .value("Translate", py::with_exception_policy::translate)
        .value("Cascade", py::with_exception_policy::cascade);

    m.def("no_args", [](const py::object &mgr) {
        py::object value;
        py::with(mgr, [&value]() {
            value = py::none();
        });
        return value;
    });

    m.def("lvalue_arg", [](const py::object &mgr) {
        py::object value;
        py::with(mgr, [&value](py::object v) {
            value = v;
        });
        return value;
    });

    m.def("lvalue_ref_arg", [](const py::object &mgr) {
        py::object value;
        py::with(mgr, [&value](py::object &v) {
            value = v;
        });
        return value;
    });

    m.def("lvalue_const_ref_arg", [](const py::object &mgr) {
        py::object value;
        py::with(mgr, [&value](const py::object &v) {
            value = v;
        });
        return value;
    });

    m.def("rvalue_ref_arg", [](const py::object &mgr) {
        py::object value;
        py::with(mgr, [&value](py::object &&v) {
            value = v;
        });
        return value;
    });

    m.def("python_exception", [](const py::object &mgr, py::with_exception_policy exception_policy) {
        py::object value;
        py::with(mgr, [&value](py::object v) {
            value = v;
            py::exec("raise RuntimeError('This is a test. Please stay calm.')");
        }, exception_policy);
        return value;
    });

    m.def("cpp_exception", [](const py::object &mgr, py::with_exception_policy exception_policy) {
        py::object value;
        py::with(mgr, [&value](py::object v) {
            value = v;
            throw std::runtime_error("This is a test. Please stay calm.");
        }, exception_policy);
        return value;
    });

    m.def("catch_cpp_exception", [](const py::object &mgr) {
        try {
            py::with(mgr, []() {
                throw NewCppException();
            });
        }
        catch (const py::error_already_set &) {
            return "error_already_set";
        }
        catch (const NewCppException &) {
            return "original_exception";
        }
        catch (...) {
            return "another_exception";
        }
        return "no_exception";
    });

    m.def("SHOULD_NOT_COMPILE_UNCOMMENTED", [](const py::object &mgr) {
        // py::with(mgr, [](int) {});
        // py::with(mgr, [](py::object, int) {});
        // py::with(mgr, [](py::object) { return "something"; });
        (void) mgr;
    });
}
