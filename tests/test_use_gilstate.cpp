#include "pybind11_tests.h"

#if defined(WITH_THREAD)

#include <thread>

static bool check_internal(bool option_use_gilstate) {
    auto const &internals = py::detail::get_internals();
#if !defined(PYPY_VERSION)
    if (option_use_gilstate)
        return !PyThread_get_key_value(internals.tstate);
    else
        return !!PyThread_get_key_value(internals.tstate);
#else
    (void)option_use_gilstate;
    return !PyThread_get_key_value(internals.tstate);
#endif
}

TEST_SUBMODULE(use_gilstate, m) {
    m.def("check_use_gilstate", [](bool option_use_gilstate) -> bool {
        py::options options;
        if (option_use_gilstate)
            options.enable_use_gilstate();
        else
            options.disable_use_gilstate();
        {
            py::gil_scoped_acquire acquire;
            return check_internal(option_use_gilstate);
        }
    }, py::call_guard<py::gil_scoped_release>())
    .def("check_default", []() -> bool {
        py::gil_scoped_acquire acquire;
        return check_internal(false);
    }, py::call_guard<py::gil_scoped_release>())
    .def("check_use_gilstate_cthread", [](bool option_use_gilstate) -> bool {
        py::options options;
        if (option_use_gilstate)
            options.enable_use_gilstate();
        else
            options.disable_use_gilstate();

        bool result = false;
        std::thread thread([option_use_gilstate, &result]() {
            py::gil_scoped_acquire acquire;
            result = check_internal(option_use_gilstate);
        });
        thread.join();
        return result;
    }, py::call_guard<py::gil_scoped_release>());
}

#endif
