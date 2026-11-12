/*
    pybind11/eval.h: Support for evaluating Python expressions and statements
    from strings and files

    Copyright (c) 2016 Klemens Morgenstern <klemens.morgenstern@ed-chemnitz.de> and
                       Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "pybind11.h"

#include <utility>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

inline void ensure_builtins_in_globals(object &global) {
#if defined(PYPY_VERSION) || defined(Py_LIMITED_API)
    // Running exec and eval adds `builtins` module under `__builtins__` key to
    // globals if not yet present.  Python 3.8 made PyRun_String behave
    // similarly. Let's also do that for older versions, for consistency. This
    // was missing from PyPy3.8 7.3.7, and PyEval_EvalCode (the stable-ABI path
    // below) does not do it either.
    if (!global.contains("__builtins__"))
        global["__builtins__"] = module_::import(PYBIND11_BUILTINS_MODULE);
#else
    (void) global;
#endif
}

#if defined(Py_LIMITED_API)
// PyRun_String() is not part of the stable ABI: compile and evaluate in two steps.
inline PyObject *run_string(
    const char *source, const char *filename, int start, PyObject *global, PyObject *local) {
    auto code = reinterpret_steal<object>(Py_CompileString(source, filename, start));
    if (!code) {
        return nullptr;
    }
    return PyEval_EvalCode(code.ptr(), global, local);
}
#else
inline PyObject *run_string(
    const char *source, const char * /*filename*/, int start, PyObject *global, PyObject *local) {
    return PyRun_String(source, start, global, local);
}
#endif

PYBIND11_NAMESPACE_END(detail)

enum eval_mode {
    /// Evaluate a string containing an isolated expression
    eval_expr,

    /// Evaluate a string containing a single statement. Returns \c none
    eval_single_statement,

    /// Evaluate a string containing a sequence of statement. Returns \c none
    eval_statements
};

template <eval_mode mode = eval_expr>
object eval(const str &expr, object global = globals(), object local = object()) {
    if (!local) {
        local = global;
    }

    detail::ensure_builtins_in_globals(global);

    // PyRun_String requires a plain C string; Python 3 always assumes UTF-8 source.
    std::string buffer = (std::string) expr;

    int start = 0;
    switch (mode) {
        case eval_expr:
            start = Py_eval_input;
            break;
        case eval_single_statement:
            start = Py_single_input;
            break;
        case eval_statements:
            start = Py_file_input;
            break;
        default:
            pybind11_fail("invalid evaluation mode");
    }

    PyObject *result
        = detail::run_string(buffer.c_str(), "<string>", start, global.ptr(), local.ptr());
    if (!result) {
        throw error_already_set();
    }
    return reinterpret_steal<object>(result);
}

template <eval_mode mode = eval_expr, size_t N>
object eval(const char (&s)[N], object global = globals(), object local = object()) {
    /* Support raw string literals by removing common leading whitespace */
    auto expr = (s[0] == '\n') ? str(module_::import("textwrap").attr("dedent")(s)) : str(s);
    return eval<mode>(expr, std::move(global), std::move(local));
}

inline void exec(const str &expr, object global = globals(), object local = object()) {
    eval<eval_statements>(expr, std::move(global), std::move(local));
}

template <size_t N>
void exec(const char (&s)[N], object global = globals(), object local = object()) {
    eval<eval_statements>(s, std::move(global), std::move(local));
}

#if defined(PYPY_VERSION) || defined(GRAALVM_PYTHON)
template <eval_mode mode = eval_statements>
object eval_file(str, object, object) {
    pybind11_fail("eval_file not supported in this interpreter. Use eval");
}
template <eval_mode mode = eval_statements>
object eval_file(str, object) {
    pybind11_fail("eval_file not supported in this interpreter. Use eval");
}
template <eval_mode mode = eval_statements>
object eval_file(str) {
    pybind11_fail("eval_file not supported in this interpreter. Use eval");
}
#else
template <eval_mode mode = eval_statements>
object eval_file(str fname, object global = globals(), object local = object()) {
    if (!local) {
        local = global;
    }

    detail::ensure_builtins_in_globals(global);

    int start = 0;
    switch (mode) {
        case eval_expr:
            start = Py_eval_input;
            break;
        case eval_single_statement:
            start = Py_single_input;
            break;
        case eval_statements:
            start = Py_file_input;
            break;
        default:
            pybind11_fail("invalid evaluation mode");
    }

    std::string fname_str = (std::string) fname;
#    if defined(Py_LIMITED_API)
    // PyRun_FileEx() and Py_fopen() are not part of the stable ABI: read the file from Python.
    object source;
    try {
        object io_open = module_::import("io").attr("open");
        object file = io_open(fname, "rb");
        source = file.attr("read")();
        file.attr("close")();
    } catch (error_already_set &) {
        pybind11_fail("File \"" + fname_str + "\" could not be opened!");
    }
    if (!global.contains("__file__")) {
        global["__file__"] = std::move(fname);
    }
    PyObject *result = detail::run_string(
        PyBytes_AsString(source.ptr()), fname_str.c_str(), start, global.ptr(), local.ptr());
#    else
    int closeFile = 1;
    FILE *f =
#        if PY_VERSION_HEX >= 0x030E0000
        Py_fopen(fname.ptr(), "r");
#        else
        _Py_fopen_obj(fname.ptr(), "r");
#        endif
    if (!f) {
        PyErr_Clear();
        pybind11_fail("File \"" + fname_str + "\" could not be opened!");
    }

    if (!global.contains("__file__")) {
        global["__file__"] = std::move(fname);
    }

    PyObject *result
        = PyRun_FileEx(f, fname_str.c_str(), start, global.ptr(), local.ptr(), closeFile);
#    endif

    if (!result) {
        throw error_already_set();
    }
    return reinterpret_steal<object>(result);
}
#endif

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
