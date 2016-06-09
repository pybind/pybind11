/*
    example/example18.cpp -- Usage of exec, eval etc.

    Copyright (c) 2016 Klemens D. Morgenstern

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/


#include <pybind11/exec.h>
#include "example.h"

void example18() {
    py::module main_module = py::module::import("__main__");
    py::object main_namespace = main_module.attr("__dict__");

    bool executed = false;

    main_module.def("call_test", [&]()-> int {executed = true; return 42;});

    cout << "exec test" << endl;

    py::exec(
            "print('Hello World!');\n"
            "x = call_test();",
                          main_namespace);

    if (executed)
        cout << "exec passed" << endl;
    else {
        cout << "exec failed" << endl;
    }

    cout << "eval test" << endl;

    py::object val = py::eval("x", main_namespace);

    if (val.cast<int>() == 42)
        cout << "eval passed" << endl;
    else {
        cout << "eval failed" << endl;
    }


    executed = false;
    cout << "exec_statement test" << endl;

    py::exec_statement("y = call_test();", main_namespace);


    if (executed)
        cout << "exec_statement passed" << endl;
    else {
        cout << "exec_statement failed" << endl;
    }

    cout << "exec_file test" << endl;

    int val_out;
    main_module.def("call_test2", [&](int value) {val_out = value;});


    py::exec_file("example18_call.py", main_namespace);

    if (val_out == 42)
        cout << "exec_file passed" << endl;
    else {
        cout << "exec_file failed" << endl;
    }

    executed = false;
    cout << "exec failure test" << endl;
    try {
    	py::exec("non-sense code ...");
    }
    catch (py::error_already_set & err) {
    	executed = true;
    }
    if (executed)
        cout << "exec failure test passed" << endl;
    else {
        cout << "exec failure test failed" << endl;
    }


    executed = false;
    cout << "exec_file failure test" << endl;
    try {
    	py::exec_file("none-existing file");
    }
    catch (std::invalid_argument & err) {
    	executed = true;
    }
    if (executed)
        cout << "exec_file failure test passed" << endl;
    else {
        cout << "exec_file failure test failed" << endl;
    }

    executed = false;
    cout << "eval failure test" << endl;
    try {
    	py::eval("print('dummy')");
    }
    catch (py::error_already_set & err) {
    	executed = true;
    }
    if (executed)
        cout << "eval failure test passed" << endl;
    else {
        cout << "eval failure test failed" << endl;
    }
}

void init_ex18(py::module & m) {
	m.def("example18", &example18);
}


