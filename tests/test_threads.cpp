#include "pybind11_tests.h"

#if defined(WITH_THREAD)

#include <thread>

static bool check_threadstate() {
    return PyGILState_GetThisThreadState() == PyThreadState_Get();
}

TEST_SUBMODULE(threads, m) {
    m.def("check_pythread", []() -> bool {
        py::gil_scoped_acquire acquire1;
        return check_threadstate();
    }, py::call_guard<py::gil_scoped_release>())
    .def("check_cthread", []() -> bool {
        bool result = false;
        std::thread thread([&result]() {
            py::gil_scoped_acquire acquire;
            result = check_threadstate();
        });
        thread.join();
        return result;
    }, py::call_guard<py::gil_scoped_release>());
}

#endif
