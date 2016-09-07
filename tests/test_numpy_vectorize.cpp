/*
    tests/test_numpy_vectorize.cpp -- auto-vectorize functions over NumPy array
    arguments

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include <pybind11/numpy.h>

double my_func(int x, float y, double z) {
    py::print("my_func(x:int={}, y:float={:.0f}, z:float={:.0f})"_s.format(x, y, z));
    return (float) x*y*z;
}

std::complex<double> my_func3(std::complex<double> c) {
    return c * std::complex<double>(2.f);
}

test_initializer numpy_vectorize([](py::module &m) {
    // Vectorize all arguments of a function (though non-vector arguments are also allowed)
    m.def("vectorized_func", py::vectorize(my_func));

    // Vectorize a lambda function with a capture object (e.g. to exclude some arguments from the vectorization)
    m.def("vectorized_func2",
        [](py::array_t<int> x, py::array_t<float> y, float z) {
            return py::vectorize([z](int x, float y) { return my_func(x, y, z); })(x, y);
        }
    );

    // Vectorize a complex-valued function
    m.def("vectorized_func3", py::vectorize(my_func3));

    /// Numpy function which only accepts specific data types
    m.def("selective_func", [](py::array_t<int, py::array::c_style>) { return "Int branch taken."; });
    m.def("selective_func", [](py::array_t<float, py::array::c_style>) { return "Float branch taken."; });
    m.def("selective_func", [](py::array_t<std::complex<float>, py::array::c_style>) { return "Complex float branch taken."; });
});
