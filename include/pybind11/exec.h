/*
    pybind11/exec.h: Functions to execute python from C++. Blatantly stolen from boost.python.

    Copyright (c) 2016 Klemens D. Morgenstern <klemens.morgenstern@gmx.net>

    This code is based on the boost.python implementation, so a different license applies to this file.

    Boost Software License - Version 1.0 - August 17th, 2003

	Permission is hereby granted, free of charge, to any person or organization
	obtaining a copy of the software and accompanying documentation covered by
	this license (the "Software") to use, reproduce, display, distribute,
	execute, and transmit the Software, and to prepare derivative works of the
	Software, and to permit third-parties to whom the Software is furnished to
	do so, all subject to the following:

	The copyright notices in the Software and this entire statement, including
	the above license grant, this restriction and the following disclaimer,
	must be included in all copies of the Software, in whole or in part, and
	all derivative works of the Software, unless such copies or derivative
	works are solely in the form of machine-executable object code generated by
	a source language processor.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
	SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
	FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
	ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
	DEALINGS IN THE SOFTWARE.

*/

#pragma once

#include "pytypes.h"

NAMESPACE_BEGIN(pybind11)



inline object eval (str string, object global = object(), object local = object()) {
    if (!global) {
        if (PyObject *g = PyEval_GetGlobals())
            global = object(g, true);
        else
            global = dict();
    }
    if (!local)
        local = global;

    auto st = static_cast<std::string>(string);
    PyObject *res = PyRun_String(st.c_str() , Py_eval_input, global.ptr(), local.ptr());

    if (res == nullptr)
        throw error_already_set();

    return {res, false};
}

inline object exec (str string, object global = object(), object local = object()) {
    if (!global) {
        if (PyObject *g = PyEval_GetGlobals())
            global = object(g, true);
        else
            global = dict();
    }
    if (!local)
        local = global;

    auto st = static_cast<std::string>(string);
    PyObject *res = PyRun_String(st.c_str() , Py_file_input, global.ptr(), local.ptr());

    if (res == nullptr)
        throw error_already_set();

    return {res, false};
}

inline object exec_statement (str string, object global = object(), object local = object()) {
    if (!global) {
        if (PyObject *g = PyEval_GetGlobals())
            global = object(g, true);
        else
            global = dict();
    }
    if (!local)
        local = global;

    auto st = static_cast<std::string>(string);
    PyObject *res = PyRun_String(st.c_str() , Py_single_input, global.ptr(), local.ptr());
    if (res == nullptr)
        throw error_already_set();

    return {res, false};
}

inline object exec_file(str filename, object global = object(), object local = object()) {
    // Set suitable default values for global and local dicts.
    if (!global) {
    if (PyObject *g = PyEval_GetGlobals())
        global = object(g, true);
    else
        global = dict();
    }
    if (!local) local = global;
    // should be 'char const *' but older python versions don't use 'const' yet.

    auto f = static_cast<std::string>(filename);

    // Let python open the file to avoid potential binary incompatibilities.
#if PY_VERSION_HEX >= 0x03040000
    constexpr static int close_it = 1;
    FILE *fs = _Py_fopen(f.c_str(), "r");
#elif PY_VERSION_HEX >= 0x03000000
    constexpr static int close_it = 1;
    PyObject *fo = Py_BuildValue("s", f.c_str());
    FILE *fs = _Py_fopen(fo, "r");
    Py_DECREF(fo);
#else
    const static int close_it = 0;
    PyObject *pyfile = PyFile_FromString(&f.front(), const_cast<char*>("r"));
    if (!pyfile)
    	throw std::invalid_argument(std::string(f) + " : no such file");
    object file(pyfile, false);
    FILE *fs = PyFile_AsFile(file.ptr());
#endif
    if (fs == nullptr)
    	throw std::invalid_argument(std::string(f) + " : could not be opened");

    PyObject* res = PyRun_FileEx(fs,
                f.c_str(),
                Py_file_input,
                global.ptr(), local.ptr(),
				close_it);

    if (res == nullptr)
        throw error_already_set();

    return {res, false};

}

inline object exec (const std::string &string, object global = object(), object local = object()) {
    return exec(str(string), global, local);
}

inline object eval (const std::string & string, object global = object(), object local = object()) {
    return eval(str(string), global, local);
}

inline object exec_file(const std::string & filename, object global = object(), object local = object()) {
    return exec_file(str(filename), global, local);
}
inline object exec_statement (const std::string & string, object global = object(), object local = object()) {
    return exec_statement(str(string), global, local);
}

NAMESPACE_END(pybind11)
