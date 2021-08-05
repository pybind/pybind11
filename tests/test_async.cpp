/*
    tests/test_async.cpp -- __await__ support

    Copyright (c) 2019 Google Inc.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/
#include <chrono>
#include <thread>

#include <pybind11/pybind11.h>
#include <pybind11/async.h>

#include "pybind11_tests.h"

namespace py = pybind11;
using namespace py::literals;
using namespace std::chrono_literals;

class AsyncClass {
    public:
        unsigned int wait_async(const unsigned int n) {
            unsigned int i;
            for (i=0; i<n; i++) {
                std::this_thread::sleep_for(100ms);
                this->count++;
            }

            return this->count;
        }

    private:
        unsigned int count;
};

TEST_SUBMODULE(async_module, m) {
    struct DoesNotSupportAsync {};
    py::class_<DoesNotSupportAsync>(m, "DoesNotSupportAsync")
        .def(py::init<>());
    struct SupportsAsync {};
    py::class_<SupportsAsync>(m, "SupportsAsync")
        .def(py::init<>())
        .def("__await__", [](const SupportsAsync& self) -> py::object {
            static_cast<void>(self);
            py::object loop = py::module_::import("asyncio.events").attr("get_event_loop")();
            py::object f = loop.attr("create_future")();
            f.attr("set_result")(5);
            return f.attr("__await__")();
        });


    py::async::class_async<AsyncClass>(m, "AsyncClass")
        .def_async("wait_async", &AsyncClass::wait_async)
        .def(py::init());

    py::async::enable_async(m);
}
