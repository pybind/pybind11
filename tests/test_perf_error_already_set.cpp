#include "pybind11_tests.h"

namespace test_perf_error_already_set {

struct boost_python_error_already_set {
    virtual ~boost_python_error_already_set() {}
};

void pure_unwind(std::size_t num_iterations) {
    while (num_iterations) {
        try {
            throw boost_python_error_already_set();
        } catch (const boost_python_error_already_set &) {
        }
        num_iterations--;
    }
}

void err_set_unwind_err_clear(std::size_t num_iterations) {
    while (num_iterations) {
        try {
            PyErr_SetString(PyExc_RuntimeError, "");
            throw boost_python_error_already_set();
        } catch (const boost_python_error_already_set &) {
            PyErr_Clear();
        }
        num_iterations--;
    }
}

void err_set_err_clear(std::size_t num_iterations) {
    while (num_iterations) {
        PyErr_SetString(PyExc_RuntimeError, "");
        PyErr_Clear();
        num_iterations--;
    }
}

void err_set_error_already_set(std::size_t num_iterations) {
    while (num_iterations) {
        try {
            PyErr_SetString(PyExc_RuntimeError, "");
            throw py::error_already_set();
        } catch (const py::error_already_set &) {
        }
        num_iterations--;
    }
}

void err_set_err_fetch(std::size_t num_iterations) {
    PyObject *exc_type, *exc_value, *exc_trace;
    while (num_iterations) {
        PyErr_SetString(PyExc_RuntimeError, "");
        PyErr_Fetch(&exc_type, &exc_value, &exc_trace);
        num_iterations--;
    }
}

void error_already_set_restore(std::size_t num_iterations) {
    PyErr_SetString(PyExc_RuntimeError, "");
    while (num_iterations) {
        try {
            throw py::error_already_set();
        } catch (py::error_already_set &e) {
            e.restore();
        }
        num_iterations--;
    }
    PyErr_Clear();
}

void err_fetch_err_restore(std::size_t num_iterations) {
    PyErr_SetString(PyExc_RuntimeError, "");
    PyObject *exc_type, *exc_value, *exc_trace;
    while (num_iterations) {
        PyErr_Fetch(&exc_type, &exc_value, &exc_trace);
        PyErr_Restore(exc_type, exc_value, exc_trace);
        num_iterations--;
    }
    PyErr_Clear();
}

// https://github.com/pybind/pybind11/pull/1895 original PR description.
py::int_ pr1895_original_foo() {
    py::dict d;
    try {
        return d["foo"];
    } catch (const py::error_already_set &) {
        return py::int_(42);
    }
}

} // namespace test_perf_error_already_set

TEST_SUBMODULE(perf_error_already_set, m) {
    using namespace test_perf_error_already_set;
    m.def("pure_unwind", pure_unwind);
    m.def("err_set_unwind_err_clear", err_set_unwind_err_clear);
    m.def("err_set_err_clear", err_set_err_clear);
    m.def("err_set_error_already_set", err_set_error_already_set);
    m.def("err_set_err_fetch", err_set_err_fetch);
    m.def("error_already_set_restore", error_already_set_restore);
    m.def("err_fetch_err_restore", err_fetch_err_restore);
    m.def("pr1895_original_foo", pr1895_original_foo);
}
