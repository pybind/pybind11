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

namespace test_scoped_critical_section_ns {

void test_one_nullptr() { py::scoped_critical_section lock{py::handle{}}; }

void test_two_nullptrs() { py::scoped_critical_section lock{py::handle{}, py::handle{}}; }

void test_first_nullptr() {
    py::dict d;
    py::scoped_critical_section lock{py::handle{}, d};
}

void test_second_nullptr() {
    py::dict d;
    py::scoped_critical_section lock{d, py::handle{}};
}

// Referenced test implementation: https://github.com/PyO3/pyo3/blob/v0.25.0/src/sync.rs#L874
class BoolWrapper {
public:
    explicit BoolWrapper(bool value) : value_{value} {}
    bool get() const { return value_.load(std::memory_order_acquire); }
    void set(bool value) { value_.store(value, std::memory_order_release); }

private:
    std::atomic_bool value_{false};
};

#if defined(PYBIND11_HAS_BARRIER)

// Modifying the C/C++ members of a Python object from multiple threads requires a critical section
// to ensure thread safety and data integrity.
// These tests use a scoped critical section to ensure that the Python object is accessed in a
// thread-safe manner.

void test_scoped_critical_section(const py::handle &cls) {
    auto barrier = std::barrier(2);
    auto bool_wrapper = cls(false);
    bool output = false;

    {
        // Release the GIL to allow run threads in parallel.
        py::gil_scoped_release gil_release{};

        std::thread t1([&]() {
            // Use gil_scoped_acquire to ensure we have a valid Python thread state
            // before entering the critical section. Otherwise, the critical section
            // will cause a segmentation fault.
            py::gil_scoped_acquire ensure_tstate{};
            // Enter the critical section with the same object as the second thread.
            py::scoped_critical_section lock{bool_wrapper};
            // At this point, the object is locked by this thread via the scoped_critical_section.
            // This barrier will ensure that the second thread waits until this thread has released
            // the critical section before proceeding.
            barrier.arrive_and_wait();
            // Sleep for a short time to simulate some work in the critical section.
            // This sleep is necessary to test the locking mechanism properly.
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            auto *bw = bool_wrapper.cast<BoolWrapper *>();
            bw->set(true);
        });

        std::thread t2([&]() {
            // This thread will wait until the first thread has entered the critical section due to
            // the barrier.
            barrier.arrive_and_wait();
            {
                // Use gil_scoped_acquire to ensure we have a valid Python thread state
                // before entering the critical section. Otherwise, the critical section
                // will cause a segmentation fault.
                py::gil_scoped_acquire ensure_tstate{};
                // Enter the critical section with the same object as the first thread.
                py::scoped_critical_section lock{bool_wrapper};
                // At this point, the critical section is released by the first thread, the value
                // is set to true.
                auto *bw = bool_wrapper.cast<BoolWrapper *>();
                output = bw->get();
            }
        });

        t1.join();
        t2.join();
    }

    if (!output) {
        throw std::runtime_error("Scoped critical section test failed: output is false");
    }
}

void test_scoped_critical_section2(const py::handle &cls) {
    auto barrier = std::barrier(3);
    auto bool_wrapper1 = cls(false);
    auto bool_wrapper2 = cls(false);
    std::pair<bool, bool> output{false, false};

    {
        // Release the GIL to allow run threads in parallel.
        py::gil_scoped_release gil_release{};

        std::thread t1([&]() {
            // Use gil_scoped_acquire to ensure we have a valid Python thread state
            // before entering the critical section. Otherwise, the critical section
            // will cause a segmentation fault.
            py::gil_scoped_acquire ensure_tstate{};
            // Enter the critical section with two different objects.
            // This will ensure that the critical section is locked for both objects.
            py::scoped_critical_section lock{bool_wrapper1, bool_wrapper2};
            // At this point, objects are locked by this thread via the scoped_critical_section.
            // This barrier will ensure that other threads wait until this thread has released
            // the critical section before proceeding.
            barrier.arrive_and_wait();
            // Sleep for a short time to simulate some work in the critical section.
            // This sleep is necessary to test the locking mechanism properly.
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            auto *bw1 = bool_wrapper1.cast<BoolWrapper *>();
            auto *bw2 = bool_wrapper2.cast<BoolWrapper *>();
            bw1->set(true);
            bw2->set(true);
        });

        std::thread t2([&]() {
            // This thread will wait until the first thread has entered the critical section due to
            // the barrier.
            barrier.arrive_and_wait();
            {
                // Use gil_scoped_acquire to ensure we have a valid Python thread state
                // before entering the critical section. Otherwise, the critical section
                // will cause a segmentation fault.
                py::gil_scoped_acquire ensure_tstate{};
                // Enter the critical section with the same object as the first thread.
                py::scoped_critical_section lock{bool_wrapper1};
                // At this point, the critical section is released by the first thread, the value
                // is set to true.
                auto *bw1 = bool_wrapper1.cast<BoolWrapper *>();
                output.first = bw1->get();
            }
        });

        std::thread t3([&]() {
            // This thread will wait until the first thread has entered the critical section due to
            // the barrier.
            barrier.arrive_and_wait();
            {
                // Use gil_scoped_acquire to ensure we have a valid Python thread state
                // before entering the critical section. Otherwise, the critical section
                // will cause a segmentation fault.
                py::gil_scoped_acquire ensure_tstate{};
                // Enter the critical section with the same object as the first thread.
                py::scoped_critical_section lock{bool_wrapper2};
                // At this point, the critical section is released by the first thread, the value
                // is set to true.
                auto *bw2 = bool_wrapper2.cast<BoolWrapper *>();
                output.second = bw2->get();
            }
        });

        t1.join();
        t2.join();
        t3.join();
    }

    if (!output.first || !output.second) {
        throw std::runtime_error(
            "Scoped critical section test with two objects failed: output is false");
    }
}

void test_scoped_critical_section2_same_object_no_deadlock(const py::handle &cls) {
    auto barrier = std::barrier(2);
    auto bool_wrapper = cls(false);
    bool output = false;

    {
        // Release the GIL to allow run threads in parallel.
        py::gil_scoped_release gil_release{};

        std::thread t1([&]() {
            // Use gil_scoped_acquire to ensure we have a valid Python thread state
            // before entering the critical section. Otherwise, the critical section
            // will cause a segmentation fault.
            py::gil_scoped_acquire ensure_tstate{};
            // Enter the critical section with the same object as the second thread.
            py::scoped_critical_section lock{bool_wrapper, bool_wrapper}; // same object used here
            // At this point, the object is locked by this thread via the scoped_critical_section.
            // This barrier will ensure that the second thread waits until this thread has released
            // the critical section before proceeding.
            barrier.arrive_and_wait();
            // Sleep for a short time to simulate some work in the critical section.
            // This sleep is necessary to test the locking mechanism properly.
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            auto *bw = bool_wrapper.cast<BoolWrapper *>();
            bw->set(true);
        });

        std::thread t2([&]() {
            // This thread will wait until the first thread has entered the critical section due to
            // the barrier.
            barrier.arrive_and_wait();
            {
                // Use gil_scoped_acquire to ensure we have a valid Python thread state
                // before entering the critical section. Otherwise, the critical section
                // will cause a segmentation fault.
                py::gil_scoped_acquire ensure_tstate{};
                // Enter the critical section with the same object as the first thread.
                py::scoped_critical_section lock{bool_wrapper};
                // At this point, the critical section is released by the first thread, the value
                // is set to true.
                auto *bw = bool_wrapper.cast<BoolWrapper *>();
                output = bw->get();
            }
        });

        t1.join();
        t2.join();
    }

    if (!output) {
        throw std::runtime_error(
            "Scoped critical section test with same object failed: output is false");
    }
}

#else

void test_scoped_critical_section(const py::handle &) {}
void test_scoped_critical_section2(const py::handle &) {}
void test_scoped_critical_section2_same_object_no_deadlock(const py::handle &) {}

#endif

} // namespace test_scoped_critical_section_ns

TEST_SUBMODULE(scoped_critical_section, m) {
    using namespace test_scoped_critical_section_ns;

    m.def("test_one_nullptr", test_one_nullptr);
    m.def("test_two_nullptrs", test_two_nullptrs);
    m.def("test_first_nullptr", test_first_nullptr);
    m.def("test_second_nullptr", test_second_nullptr);

    auto BoolWrapperClass = py::class_<BoolWrapper>(m, "BoolWrapper")
                                .def(py::init<bool>())
                                .def("get", &BoolWrapper::get)
                                .def("set", &BoolWrapper::set);
    auto BoolWrapperHandle = py::handle(BoolWrapperClass);
    (void) BoolWrapperHandle.ptr(); // suppress unused variable warning

    m.attr("has_barrier") =
#ifdef PYBIND11_HAS_BARRIER
        true;
#else
        false;
#endif

    m.def("test_scoped_critical_section",
          [BoolWrapperHandle]() -> void { test_scoped_critical_section(BoolWrapperHandle); });
    m.def("test_scoped_critical_section2",
          [BoolWrapperHandle]() -> void { test_scoped_critical_section2(BoolWrapperHandle); });
    m.def("test_scoped_critical_section2_same_object_no_deadlock", [BoolWrapperHandle]() -> void {
        test_scoped_critical_section2_same_object_no_deadlock(BoolWrapperHandle);
    });
}
