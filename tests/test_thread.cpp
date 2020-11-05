/*
    tests/test_tread.cpp -- thread handling

    Copyright (c) 2020 Nick Bridge <nick.bridge.chess@gmail.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include <thread>
#include <atomic>
#include <iostream>
#include <pybind11/iostream.h>


TEST_SUBMODULE(test_threading, m) {
  // object to manage C++ thread
  // simply repeatedly write to std::cerr until stopped
  // redirect is called at some point to test the safety of scoped_estream_redirect
  struct TestThread {
    TestThread() : t_{nullptr}, stop_{false} {
      auto thread_f = [this] {
        while( !this->stop_ ) {
          std::cout << "x" << std::flush;
          std::this_thread::sleep_for(std::chrono::microseconds(50));
        } };
      t_ = new std::thread(std::move(thread_f));
    }
    ~TestThread() {
      delete t_;
    }
    void stop() { stop_ = 1; }
    void join() {
      py::gil_scoped_release gil_lock;
      t_->join();
    }
    void sleep() {
      py::gil_scoped_release gil_lock;
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    std::thread * t_;
    std::atomic<bool> stop_;
  };

  py::class_<TestThread>(m, "TestThread")
      .def(py::init<>())
      .def("stop", &TestThread::stop)
      .def("join", &TestThread::join)
      .def("sleep", &TestThread::sleep);

}

