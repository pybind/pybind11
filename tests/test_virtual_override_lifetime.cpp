// Copyright (c) 2026 The pybind Community.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pybind11_tests.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <utility>

namespace {

struct Observer {
    virtual ~Observer() = default;
    virtual int on_event() { return 1; }
};

struct PyObserver : Observer {
    int on_event() override { PYBIND11_OVERRIDE(int, Observer, on_event, ); }
};

// A C++ dispatcher keeps its observer alive independently of any Python views of it.
class Dispatcher {
public:
    explicit Dispatcher(std::shared_ptr<Observer> observer) : observer_(std::move(observer)) {}

    void set_observer(std::shared_ptr<Observer> observer) {
        std::lock_guard<std::mutex> lock(mutex_);
        observer_ = std::move(observer);
    }

    int notify() {
        std::shared_ptr<Observer> observer;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            observer = observer_;
        }
        return observer->on_event();
    }

    std::size_t run() {
        std::size_t calls = 0;
        while (!stopped_.load(std::memory_order_relaxed)) {
            notify();
            ++calls;
        }
        return calls;
    }

    void stop() { stopped_.store(true); }

private:
    std::mutex mutex_;
    std::shared_ptr<Observer> observer_;
    std::atomic<bool> stopped_{false};
};

} // namespace

TEST_SUBMODULE(virtual_override_lifetime, m) {
    py::class_<Observer, PyObserver, std::shared_ptr<Observer>>(m, "Observer")
        .def(py::init<>())
        .def("on_event", &Observer::on_event);
    py::class_<Dispatcher>(m, "Dispatcher")
        .def(py::init<std::shared_ptr<Observer>>())
        .def("set_observer", &Dispatcher::set_observer)
        .def("notify", &Dispatcher::notify)
        .def("run", &Dispatcher::run, py::call_guard<py::gil_scoped_release>())
        .def("stop", &Dispatcher::stop);
}
