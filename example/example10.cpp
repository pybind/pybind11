/*
    example/example10.cpp -- Example 10: auto-vectorize functions for NumPy

    Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"
#include <pybind/numpy.h>

double my_func(int x, float y, double z) {
    std::cout << "my_func(x:int=" << x << ", y:float=" << y << ", z:float=" << z << ")" << std::endl;
    return x*y*z;
}

void init_ex10(py::module &m) {
    // Vectorize all arguments (though non-vector arguments are also allowed)
    m.def("vectorized_func", py::vectorize(my_func));

    // Vectorize a lambda function with a capture object (e.g. to exclude some arguments from the vectorization)
    m.def("vectorized_func2",
        [](py::array_dtype<int> x, py::array_dtype<float> y, float z) {
            return py::vectorize([z](int x, float y) { return my_func(x, y, z); })(x, y);
        }
    );
}
