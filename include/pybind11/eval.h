/*
    pybind11/exec.h: Support for evaluating Python expressions and statements
    from strings and files

    Copyright (c) 2016 Klemens Morgenstern <klemens.morgenstern@ed-chemnitz.de> and
                       Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#pragma once

#include "pytypes.h"

NAMESPACE_BEGIN(pybind11)

enum eval_mode {
    /// Evaluate a string containing an isolated expression
    eval_expr,

    /// Evaluate a string containing a single statement. Returns \c none
    eval_single_statement,

    /// Evaluate a string containing a sequence of statement. Returns \c none
    eval_statements
};

template <eval_mode mode = eval_expr>
object eval(const std::string& str, object global = object(), object local = object()) {
    if (!global) {
        global = object(PyEval_GetGlobals(), true);
        if (!global)
            global = dict();
    }
    if (!local)
        local = global;

    int start;
    switch (mode) {
        case eval_expr:             start = Py_eval_input;   break;
        case eval_single_statement: start = Py_single_input; break;
        case eval_statements:       start = Py_file_input;   break;
        default: pybind11_fail("invalid evaluation mode");
    }

    object result(PyRun_String(str.c_str(), start, global.ptr(), local.ptr()), false);

    if (!result)
        throw error_already_set();
    return result;
}

template <eval_mode mode = eval_statements>
object eval_file(const std::string& fname, object global = object(), object local = object()) {
    if (!global) {
        global = object(PyEval_GetGlobals(), true);
        if (!global)
            global = dict();
    }
    if (!local)
        local = global;

    int start;
    switch (mode) {
        case eval_expr:             start = Py_eval_input;   break;
        case eval_single_statement: start = Py_single_input; break;
        case eval_statements:       start = Py_file_input;   break;
        default: pybind11_fail("invalid evaluation mode");
    }

    FILE *f = fopen(fname.c_str(), "r");
    if (!f)
        pybind11_fail("File \"" + fname + "\" could not be opened!");

    object result(PyRun_FileEx(f, fname.c_str(), Py_file_input, global.ptr(),
                               local.ptr(), 1),
                  false);

    if (!result)
        throw error_already_set();

    return result;
}

NAMESPACE_END(pybind11)
