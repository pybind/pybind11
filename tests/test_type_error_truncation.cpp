/*
    tests/test_type_error_truncation.cpp -- exception translation

    Copyright (c) 2017 Thorsten Beier <thorsten.beier@iwr.uni-heidelberg.de>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include <string>


// A type that should be raised as an exeption in Python
struct TypeWithLongRepr {
    TypeWithLongRepr(const uint32_t reprSize = 100)
    : reprSize_(reprSize){

    }
    std::string repr()const{
        std::string ret(reprSize_,'*');
        ret[0] = '<';
        ret[reprSize_-1] = '>';
        return ret;
    }
    void foo(const TypeWithLongRepr & , const std::string  )const{

    }
    uint32_t reprSize_;
};



test_initializer type_error_truncation([](py::module &m) {

    py::class_<TypeWithLongRepr>(m, "TypeWithLongRepr")
        .def(py::init<const uint32_t >())
        .def("__repr__", &TypeWithLongRepr::repr)
        .def("foo",&TypeWithLongRepr::foo,
            py::arg("arg1"),
            py::arg("arg2")
        )
    ;
  
});
