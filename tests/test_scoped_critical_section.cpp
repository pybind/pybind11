#include <pybind11/critical_section.h>

#include "pybind11_tests.h"

#include <atomic>
#include <cassert>
#include <thread>

#if defined(__has_include) && __has_include(<barrier>)
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
    std::atomic<bool> value_;
};

#ifdef PYBIND11_HAS_BARRIER
void test_scoped_critical_section(py::class_<BoolWrapper> &cls) {
    auto barrier = std::barrier(2);
    auto bool_wrapper = cls(false);

    std::thread t1([&]() {
        py::scoped_critical_section lock{bool_wrapper};
        barrier.arrive_and_wait();
        auto bw = bool_wrapper.cast<std::shared_ptr<BoolWrapper>>();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        bw->set(true);
    });

    std::thread t2([&]() {
        barrier.arrive_and_wait();
        py::scoped_critical_section lock{bool_wrapper};
        auto bw = bool_wrapper.cast<std::shared_ptr<BoolWrapper>>();
        assert(bw->get() == true);
    });

    t1.join();
    t2.join();
}

void test_scoped_critical_section2(py::class_<BoolWrapper> &cls) {
    auto barrier = std::barrier(3);
    auto bool_wrapper1 = cls(false);
    auto bool_wrapper2 = cls(false);

    std::thread t1([&]() {
        py::scoped_critical_section lock{bool_wrapper1, bool_wrapper2};
        barrier.arrive_and_wait();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto bw1 = bool_wrapper1.cast<std::shared_ptr<BoolWrapper>>();
        auto bw2 = bool_wrapper2.cast<std::shared_ptr<BoolWrapper>>();
        bw1->set(true);
        bw2->set(true);
    });

    std::thread t2([&]() {
        barrier.arrive_and_wait();
        py::scoped_critical_section lock{bool_wrapper1};
        auto bw1 = bool_wrapper1.cast<std::shared_ptr<BoolWrapper>>();
        assert(bw1->get() == true);
    });

    std::thread t3([&]() {
        barrier.arrive_and_wait();
        py::scoped_critical_section lock{bool_wrapper2};
        auto bw2 = bool_wrapper2.cast<std::shared_ptr<BoolWrapper>>();
        assert(bw2->get() == true);
    });

    t1.join();
    t2.join();
    t3.join();
}

void test_scoped_critical_section2_same_object_no_deadlock(py::class_<BoolWrapper> &cls) {
    auto barrier = std::barrier(2);
    auto bool_wrapper = cls(false);

    std::thread t1([&]() {
        py::scoped_critical_section lock{bool_wrapper, bool_wrapper};
        barrier.arrive_and_wait();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto bw = bool_wrapper.cast<std::shared_ptr<BoolWrapper>>();
        bw->set(true);
    });

    std::thread t2([&]() {
        barrier.arrive_and_wait();
        py::scoped_critical_section lock{bool_wrapper};
        auto bw = bool_wrapper.cast<std::shared_ptr<BoolWrapper>>();
        assert(bw->get() == true);
    });

    t1.join();
    t2.join();
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

#ifdef PYBIND11_HAS_BARRIER
    m.attr("has_barrier") = true;

    m.def("test_scoped_critical_section",
          [&]() -> void { test_scoped_critical_section(BoolWrapperClass); });
    m.def("test_scoped_critical_section2",
          [&]() -> void { test_scoped_critical_section2(BoolWrapperClass); });
    m.def("test_scoped_critical_section2_same_object_no_deadlock", [&]() -> void {
        test_scoped_critical_section2_same_object_no_deadlock(BoolWrapperClass);
    });
#else
    m.attr("has_barrier") = false;
#endif
}
