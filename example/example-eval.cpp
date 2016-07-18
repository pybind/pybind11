/*
    example/example-eval.cpp -- Usage of eval() and eval_file()

    Copyright (c) 2016 Klemens D. Morgenstern

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/


#include <pybind11/eval.h>
#include "example.h"

void example_eval() {
    py::module main_module = py::module::import("__main__");
    py::object main_namespace = main_module.attr("__dict__");

    bool ok = false;

    main_module.def("call_test", [&]() -> int {
        ok = true;
        return 42;
    });

    cout << "eval_statements test" << endl;

    auto result = py::eval<py::eval_statements>(
            "print('Hello World!');\n"
            "x = call_test();", main_namespace);

    if (ok && result == py::none())
        cout << "eval_statements passed" << endl;
    else
        cout << "eval_statements failed" << endl;

    cout << "eval test" << endl;

    py::object val = py::eval("x", main_namespace);

    if (val.cast<int>() == 42)
        cout << "eval passed" << endl;
    else
        cout << "eval failed" << endl;

    ok = false;
    cout << "eval_single_statement test" << endl;

    py::eval<py::eval_single_statement>(
        "y = call_test();", main_namespace);

    if (ok)
        cout << "eval_single_statement passed" << endl;
    else
        cout << "eval_single_statement failed" << endl;

    cout << "eval_file test" << endl;

    int val_out;
    main_module.def("call_test2", [&](int value) {val_out = value;});

    try {
        result = py::eval_file("example-eval_call.py", main_namespace);
    } catch (...) {
        result = py::eval_file("example/example-eval_call.py", main_namespace);
    }

    if (val_out == 42 && result == py::none())
        cout << "eval_file passed" << endl;
    else
        cout << "eval_file failed" << endl;

    ok = false;
    cout << "eval failure test" << endl;
    try {
        py::eval("nonsense code ...");
    } catch (py::error_already_set &) {
        PyErr_Clear();
        ok = true;
    }

    if (ok)
        cout << "eval failure test passed" << endl;
    else
        cout << "eval failure test failed" << endl;

    ok = false;
    cout << "eval_file failure test" << endl;
    try {
        py::eval_file("nonexisting file");
    } catch (std::exception &) {
        ok = true;
    }

    if (ok)
        cout << "eval_file failure test passed" << endl;
    else
        cout << "eval_file failure test failed" << endl;
}

void init_ex_eval(py::module & m) {
    m.def("example_eval", &example_eval);
}
