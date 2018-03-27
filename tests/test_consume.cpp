/*
    tests/test_consume.cpp -- consume call policy

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>
    Copyright (c) 2017 Attila Török <torokati44@gmail.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

class Box {
    int size;
    static int num_boxes;

public:
    Box(int size): size(size) { py::print("Box created."); ++num_boxes; }
    ~Box() { py::print("Box destroyed."); --num_boxes; }

    int get_size() { return size; }
    static int get_num_boxes() { return num_boxes; }
};

int Box::num_boxes = 0;

class Filter {
    int threshold;

public:
    Filter(int threshold): threshold(threshold) { py::print("Filter created."); }
    ~Filter() { py::print("Filter destroyed."); }

    void process(Box *box) { // ownership of box is taken
        py::print("Box is processed by Filter.");
        if (box->get_size() > threshold)
            delete box;
        // otherwise the box is leaked
    };
};

test_initializer consume([](py::module &m) {
    py::class_<Box>(m, "Box")
        .def(py::init<int>())
        .def_static("get_num_boxes", &Box::get_num_boxes);

    py::class_<Filter>(m, "Filter")
        .def(py::init<int>())
        .def("process", &Filter::process, py::consume<2>());
});
