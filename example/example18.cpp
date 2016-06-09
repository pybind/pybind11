/*
    example/example18.cpp -- Usage of exec, eval etc.

    Copyright (c) 2016 Klemens D. Morgenstern

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"

int main(int argc, char* argv[])
{
	 Py_Initialize() ;

	py::module main_module = py::module::import("__main__");
	py::object main_namespace = main_module.attr("__dict__");

	bool errored = false;

	bool executed = false;

	main_module.def("call_test", [&]()-> int {executed = true; return 42;});

	cout << "exec test" << endl;

	py::exec(
			"print('Hello World!');\n"
			"x = call_test();",
	                      main_namespace);

	if (executed)
		cout << "exec passed" << endl;
	else
	{
		cout << "exec failed" << endl;
		errored = true;
	}

	cout << "eval test" << endl;

	py::object val = py::eval("x", main_namespace);

	if (val.cast<int>() == 42)
		cout << "eval passed" << endl;
	else
	{
		cout << "eval failed" << endl;
		errored = true;
	}


	executed = false;
	cout << "exec_statement test" << endl;

	py::exec_statement("y = call_test();", main_namespace);


	if (executed)
		cout << "exec_statement passed" << endl;
	else
	{
		cout << "exec_statement failed" << endl;
		errored = true;
	}

	cout << "exec_file test" << endl;

	int val_out;
	main_module.def("call_test2", [&](int value) {val_out = value;});


	py::exec_file("example18.py", main_namespace);

	if (val_out == 42)
		cout << "exec_file passed" << endl;
	else
	{
		cout << "exec_file failed" << endl;
		errored = true;
	}

	return errored ? 1 : 0;
}


