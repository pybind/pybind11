/*
    tests/test_thread.cpp -- call pybind11 bound methods in threads

    Copyright (c) 2021 Laramie Leavitt (Google LLC) <lar@google.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>

#include "pybind11_tests.h"

#include <chrono>
#include <thread>

#if defined(PYBIND11_HAS_STD_BARRIER)
#    include <barrier>
#endif

namespace py = pybind11;

namespace {

struct IntStruct {
    explicit IntStruct(int v) : value(v) {};
    ~IntStruct() { value = -value; }
    IntStruct(const IntStruct &) = default;
    IntStruct &operator=(const IntStruct &) = default;

    int value;
};

struct EmptyStruct {};
EmptyStruct SharedInstance;

} // namespace

TEST_SUBMODULE(thread, m) {
    py::class_<IntStruct>(m, "IntStruct").def(py::init([](const int i) { return IntStruct(i); }));

    // implicitly_convertible uses loader_life_support when an implicit
    // conversion is required in order to lifetime extend the reference.
    //
    // This test should be run with ASAN for better effectiveness.
    py::implicitly_convertible<int, IntStruct>();

    m.def("test", [](int expected, const IntStruct &in) {
        {
            py::gil_scoped_release release;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        if (in.value != expected) {
            throw std::runtime_error("Value changed!!");
        }
    });

    m.def(
        "test_no_gil",
        [](int expected, const IntStruct &in) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            if (in.value != expected) {
                throw std::runtime_error("Value changed!!");
            }
        },
        py::call_guard<py::gil_scoped_release>());

    py::class_<EmptyStruct>(m, "EmptyStruct")
        .def_readonly_static("SharedInstance", &SharedInstance);

#if defined(PYBIND11_HAS_STD_BARRIER)
    // In the free-threaded build, during PyThreadState_Clear, removing the thread from the biased
    // reference counting table may call destructors. Make sure that it doesn't crash.
    m.def("test_pythread_state_clear_destructor", [](py::type cls) {
        py::handle obj;

        std::barrier barrier{2};
        std::thread thread1{[&]() {
            py::gil_scoped_acquire gil;
            obj = cls().release();
            barrier.arrive_and_wait();
        }};
        std::thread thread2{[&]() {
            py::gil_scoped_acquire gil;
            barrier.arrive_and_wait();
            // ob_ref_shared becomes negative; transition to the queued state
            obj.dec_ref();
        }};

        // jthread is not supported by Apple Clang
        thread1.join();
        thread2.join();
    });
#endif

    m.attr("defined_PYBIND11_HAS_STD_BARRIER") =
#ifdef PYBIND11_HAS_STD_BARRIER
        true;
#else
        false;
#endif
    m.def("acquire_gil", []() { py::gil_scoped_acquire gil_acquired; });

    // NOTE: std::string_view also uses loader_life_support to ensure that
    // the string contents remain alive, but that's a C++ 17 feature.
}
