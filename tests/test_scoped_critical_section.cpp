#include <pybind11/critical_section.h>

#include "pybind11_tests.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <utility>

#if defined(PYBIND11_CPP20) && defined(__has_include) && __has_include(<barrier>)
#    define PYBIND11_HAS_BARRIER 1
#    include <barrier>
#endif

// Referenced test implementation: https://github.com/PyO3/pyo3/blob/v0.25.0/src/sync.rs#L874
class BoolWrapper {
public:
    explicit BoolWrapper(bool value) : value_{value} {}
    bool get() const { return value_.load(std::memory_order_acquire); }
    void set(bool value) { value_.store(value, std::memory_order_release); }

private:
    std::atomic<bool> value_{false};
};

#ifdef PYBIND11_HAS_BARRIER
bool test_scoped_critical_section(const py::handle &cls) {
    auto barrier = std::barrier(2);
    auto bool_wrapper = cls(false);
    bool output = false;

    std::thread t1([&]() {
        py::scoped_critical_section lock{bool_wrapper};
        barrier.arrive_and_wait();
        auto *bw = bool_wrapper.cast<BoolWrapper *>();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        bw->set(true);
    });

    std::thread t2([&]() {
        barrier.arrive_and_wait();
        py::scoped_critical_section lock{bool_wrapper};
        auto *bw = bool_wrapper.cast<BoolWrapper *>();
        output = bw->get();
    });

    t1.join();
    t2.join();

    return output;
}

std::pair<bool, bool> test_scoped_critical_section2(const py::handle &cls) {
    auto barrier = std::barrier(3);
    auto bool_wrapper1 = cls(false);
    auto bool_wrapper2 = cls(false);
    std::pair<bool, bool> output{false, false};

    std::thread t1([&]() {
        py::scoped_critical_section lock{bool_wrapper1, bool_wrapper2};
        barrier.arrive_and_wait();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto *bw1 = bool_wrapper1.cast<BoolWrapper *>();
        auto *bw2 = bool_wrapper2.cast<BoolWrapper *>();
        bw1->set(true);
        bw2->set(true);
    });

    std::thread t2([&]() {
        barrier.arrive_and_wait();
        py::scoped_critical_section lock{bool_wrapper1};
        auto *bw1 = bool_wrapper1.cast<BoolWrapper *>();
        output.first = bw1->get();
    });

    std::thread t3([&]() {
        barrier.arrive_and_wait();
        py::scoped_critical_section lock{bool_wrapper2};
        auto *bw2 = bool_wrapper2.cast<BoolWrapper *>();
        output.second = bw2->get();
    });

    t1.join();
    t2.join();
    t3.join();

    return output;
}

bool test_scoped_critical_section2_same_object_no_deadlock(const py::handle &cls) {
    auto barrier = std::barrier(2);
    auto bool_wrapper = cls(false);
    bool output = false;

    std::thread t1([&]() {
        py::scoped_critical_section lock{bool_wrapper, bool_wrapper};
        barrier.arrive_and_wait();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto *bw = bool_wrapper.cast<BoolWrapper *>();
        bw->set(true);
    });

    std::thread t2([&]() {
        barrier.arrive_and_wait();
        py::scoped_critical_section lock{bool_wrapper};
        auto *bw = bool_wrapper.cast<BoolWrapper *>();
        output = bw->get();
    });

    t1.join();
    t2.join();

    return output;
}
#endif

TEST_SUBMODULE(scoped_critical_section, m) {
    m.attr("defined_THREAD_SANITIZER") =
#if defined(THREAD_SANITIZER)
        true;
#else
        false;
#endif

    auto BoolWrapperClass = py::class_<BoolWrapper>(m, "BoolWrapper")
                                .def(py::init<bool>())
                                .def("get", &BoolWrapper::get)
                                .def("set", &BoolWrapper::set);
    auto BoolWrapperHandle = py::handle(BoolWrapperClass);
    (void) BoolWrapperHandle.ptr(); // suppress unused variable warning

#ifdef PYBIND11_HAS_BARRIER
    m.attr("has_barrier") = true;

    m.def("test_scoped_critical_section", [BoolWrapperHandle]() -> bool {
        return test_scoped_critical_section(BoolWrapperHandle);
    });
    m.def("test_scoped_critical_section2", [BoolWrapperHandle]() -> std::pair<bool, bool> {
        return test_scoped_critical_section2(BoolWrapperHandle);
    });
    m.def("test_scoped_critical_section2_same_object_no_deadlock", [BoolWrapperHandle]() -> bool {
        return test_scoped_critical_section2_same_object_no_deadlock(BoolWrapperHandle);
    });
#else
    m.attr("has_barrier") = false;
#endif
}
