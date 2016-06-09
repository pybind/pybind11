/*
    pybind11/exec.h: Functions to execute python from C++. Blatantly stolen from boost.python.

    Copyright (c) 2016 Klemens D. Morgenstern <klemens.morgenstern@gmx.net>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "pytypes.h"

NAMESPACE_BEGIN(pybind11)

inline object eval (str string, object global = object(), object local = object())
{
	if (!global)
	{
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

inline object exec (str string, object global = object(), object local = object())
{
	if (!global)
	{
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

inline object exec_statement (str string, object global = object(), object local = object())
{
	if (!global)
	{
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

object exec_file(str filename, object global = object(), object local = object())
{
  // Set suitable default values for global and local dicts.
  if (!global)
  {
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
  FILE *fs = _Py_fopen(f.c_str(), "r");
#elif PY_VERSION_HEX >= 0x03000000
  PyObject *fo = Py_BuildValue("s", f);
  FILE *fs = _Py_fopen(fo, "r");
  Py_DECREF(fo);
#else
  PyObject *pyfile = PyFile_FromString(f, const_cast<char*>("r"));
  if (!pyfile) throw std::invalid_argument(std::string(f) + " : no such file");
  python::handle<> file(pyfile);
  FILE *fs = PyFile_AsFile(file.get());
#endif
  PyObject* res = PyRun_File(fs,
                f.c_str(),
                Py_file_input,
		global.ptr(), local.ptr());

  if (res == nullptr)
		throw error_already_set();

  return {res, false};

}

NAMESPACE_END(pybind11)
