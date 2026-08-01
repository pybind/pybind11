// Shared harness for the py::print-during-shutdown regression tests in
// test_interpreter.cpp and test_subinterpreter.cpp.

#pragma once

#include <pybind11/pybind11.h>

struct print_shutdown_state {
    bool callback_ran = false;
    bool stdout_was_none = false;
    bool print_threw = false;
};

// Attach a capsule to sys whose destructor calls py::print during interpreter
// shutdown, recording what happened in `state`.
inline void install_print_shutdown_probe(print_shutdown_state &state) {
    namespace py = pybind11;
    py::module_::import("sys").attr("pybind11_print_on_shutdown")
        = py::capsule(&state, [](void *payload) noexcept {
              auto *state = static_cast<print_shutdown_state *>(payload);
              state->callback_ran = true;
              state->stdout_was_none = PySys_GetObject("stdout") == Py_None;
              try {
                  py::print("print during interpreter shutdown");
              } catch (...) {
                  state->print_threw = true;
              }
          });
}
