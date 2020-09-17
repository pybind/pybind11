/*
    tests/test_ctypes_buffer.cpp -- __await__ support

    Copyright (c) 2020 Fritz Reese <fritzoreese@gmail.com>.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

TEST_SUBMODULE(ctypes_buffer, m) {
  m.def("get_ctypes_buffer_size", [](py::buffer buffer) {
    py::buffer_info info = buffer.request();
    return info.shape[0];
  });
}
