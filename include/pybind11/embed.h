/*
    pybind11/embed.h: Support for embedding the interpreter

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "pybind11.h"
#include "eval.h"

#if defined(PYPY_VERSION)
#    error Embedding the interpreter is not supported with PyPy
#endif

#define PYBIND11_EMBEDDED_MODULE_IMPL(name)                                                       \
    extern "C" PyObject *pybind11_init_impl_##name();                                             \
    extern "C" PyObject *pybind11_init_impl_##name() { return pybind11_init_wrapper_##name(); }

/** \rst
    Add a new module to the table of builtins for the interpreter. Must be
    defined in global scope. The first macro parameter is the name of the
    module (without quotes). The second parameter is the variable which will
    be used as the interface to add functions and classes to the module.

    .. code-block:: cpp

        PYBIND11_EMBEDDED_MODULE(example, m) {
            // ... initialize functions and classes here
            m.def("foo", []() {
                return "Hello, World!";
            });
        }

    The third and subsequent macro arguments are optional, and can be used to
    mark the module as supporting various Python features.

    - ``mod_gil_not_used()``
    - ``multiple_interpreters::per_interpreter_gil()``
    - ``multiple_interpreters::shared_gil()``
    - ``multiple_interpreters::not_supported()``

    .. code-block:: cpp

        PYBIND11_EMBEDDED_MODULE(example, m, py::mod_gil_not_used()) {
            m.def("foo", []() {
                return "Hello, Free-threaded World!";
            });
        }

 \endrst */
PYBIND11_WARNING_PUSH
PYBIND11_WARNING_DISABLE_CLANG("-Wgnu-zero-variadic-macro-arguments")
#define PYBIND11_EMBEDDED_MODULE(name, variable, ...)                                             \
    PYBIND11_MODULE_PYINIT(name, ##__VA_ARGS__)                                                   \
    ::pybind11::detail::embedded_module PYBIND11_CONCAT(pybind11_module_, name)(                  \
        PYBIND11_TOSTRING(name), PYBIND11_CONCAT(PyInit_, name));                                 \
    PYBIND11_MODULE_EXEC(name, variable)
PYBIND11_WARNING_POP

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

/// Python 2.7/3.x compatible version of `PyImport_AppendInittab` and error checks.
struct embedded_module {
    using init_t = PyObject *(*) ();
    embedded_module(const char *name, init_t init) {
        if (Py_IsInitialized() != 0) {
            pybind11_fail("Can't add new modules after the interpreter has been initialized");
        }

        auto result = PyImport_AppendInittab(name, init);
        if (result == -1) {
            pybind11_fail("Insufficient memory to add a new module");
        }
    }
};

inline void precheck_interpreter() {
    if (Py_IsInitialized() != 0) {
        pybind11_fail("The interpreter is already running");
    }
}

PYBIND11_NAMESPACE_END(detail)

inline void initialize_interpreter(PyConfig *config,
                                   int argc = 0,
                                   const char *const *argv = nullptr,
                                   bool add_program_dir_to_path = true) {
    detail::precheck_interpreter();
    PyStatus status = PyConfig_SetBytesArgv(config, argc, const_cast<char *const *>(argv));
    if (PyStatus_Exception(status) != 0) {
        // A failure here indicates a character-encoding failure or the python
        // interpreter out of memory. Give up.
        PyConfig_Clear(config);
        throw std::runtime_error(PyStatus_IsError(status) != 0 ? status.err_msg
                                                               : "Failed to prepare CPython");
    }
    status = Py_InitializeFromConfig(config);
    if (PyStatus_Exception(status) != 0) {
        PyConfig_Clear(config);
        throw std::runtime_error(PyStatus_IsError(status) != 0 ? status.err_msg
                                                               : "Failed to init CPython");
    }
    if (add_program_dir_to_path) {
        PyRun_SimpleString("import sys, os.path; "
                           "sys.path.insert(0, "
                           "os.path.abspath(os.path.dirname(sys.argv[0])) "
                           "if sys.argv and os.path.exists(sys.argv[0]) else '')");
    }
    PyConfig_Clear(config);
}

/** \rst
    Initialize the Python interpreter. No other pybind11 or CPython API functions can be
    called before this is done; with the exception of `PYBIND11_EMBEDDED_MODULE`. The
    optional `init_signal_handlers` parameter can be used to skip the registration of
    signal handlers. Calling this function again while an interpreter is running throws
    ``std::runtime_error``.

    The interpreter starts from a ``PyConfig``, with ``PyConfig_SetBytesArgv`` and
    ``Py_InitializeFromConfig``. If the start fails, this function throws
    ``std::runtime_error``. See the `Python documentation`_ for these functions.

    The remaining optional parameters, `argc`, `argv`, and `add_program_dir_to_path` are
    used to populate ``sys.argv`` and ``sys.path``.

    .. _Python documentation: https://docs.python.org/3/c-api/init_config.html
 \endrst */
inline void initialize_interpreter(bool init_signal_handlers = true,
                                   int argc = 0,
                                   const char *const *argv = nullptr,
                                   bool add_program_dir_to_path = true) {
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    // See PR #4473 for background
    config.parse_argv = 0;

    config.install_signal_handlers = init_signal_handlers ? 1 : 0;
    initialize_interpreter(&config, argc, argv, add_program_dir_to_path);

    // There is exactly one interpreter alive currently.
    detail::has_seen_non_main_interpreter() = false;
}

/** \rst
    Shut down the Python interpreter. No pybind11 or CPython API functions can be called
    after this. In addition, pybind11 objects must not outlive the interpreter:

    .. code-block:: cpp

        { // BAD
            py::initialize_interpreter();
            auto hello = py::str("Hello, World!");
            py::finalize_interpreter();
        } // <-- BOOM, hello's destructor is called after interpreter shutdown

        { // GOOD
            py::initialize_interpreter();
            { // scoped
                auto hello = py::str("Hello, World!");
            } // <-- OK, hello is cleaned up properly
            py::finalize_interpreter();
        }

        { // BETTER
            py::scoped_interpreter guard{};
            auto hello = py::str("Hello, World!");
        }

    .. warning::

        The interpreter can be restarted by calling `initialize_interpreter` again.
        Modules created using pybind11 can be safely re-initialized. However, Python
        itself cannot completely unload binary extension modules and there are several
        caveats with regard to interpreter restarting. All the details can be found
        in the CPython documentation. In short, not all interpreter memory may be
        freed, either due to reference cycles or user-created global data.

 \endrst */
inline void finalize_interpreter() {
    // get rid of any thread-local interpreter cache that currently exists
    if (detail::has_seen_non_main_interpreter()) {
        detail::get_internals_pp_manager().unref();
        detail::get_local_internals_pp_manager().unref();

        // We know there can be no other interpreter alive now
        detail::has_seen_non_main_interpreter() = false;
    }

    // Re-fetch the internals pointer-to-pointer (but not the internals itself, which might not
    // exist). It's possible for the  internals to be created during Py_Finalize() (e.g. if a
    // py::capsule calls `get_internals()` during destruction), so we get the pointer-pointer here
    // and check it after Py_Finalize().
    detail::get_internals_pp_manager().get_pp();
    detail::get_local_internals_pp_manager().get_pp();

    Py_Finalize();

    detail::get_internals_pp_manager().destroy();

    // Local internals contains data managed by the current interpreter, so we must clear them to
    // avoid undefined behaviors when initializing another interpreter
    detail::get_local_internals_pp_manager().destroy();

    // We know there is no interpreter alive now, so we can reset the multi-flag
    detail::has_seen_non_main_interpreter() = false;
}

/** \rst
    Scope guard version of `initialize_interpreter` and `finalize_interpreter`.
    This a move-only guard and only a single instance can exist.

    See `initialize_interpreter` for a discussion of its constructor arguments.

    .. code-block:: cpp

        #include <pybind11/embed.h>

        int main() {
            py::scoped_interpreter guard{};
            py::print(Hello, World!);
        } // <-- interpreter shutdown
 \endrst */
class scoped_interpreter {
public:
    explicit scoped_interpreter(bool init_signal_handlers = true,
                                int argc = 0,
                                const char *const *argv = nullptr,
                                bool add_program_dir_to_path = true) {
        initialize_interpreter(init_signal_handlers, argc, argv, add_program_dir_to_path);
    }

    explicit scoped_interpreter(PyConfig *config,
                                int argc = 0,
                                const char *const *argv = nullptr,
                                bool add_program_dir_to_path = true) {
        initialize_interpreter(config, argc, argv, add_program_dir_to_path);
    }

    scoped_interpreter(const scoped_interpreter &) = delete;
    scoped_interpreter(scoped_interpreter &&other) noexcept { other.is_valid = false; }
    scoped_interpreter &operator=(const scoped_interpreter &) = delete;
    scoped_interpreter &operator=(scoped_interpreter &&) = delete;

    ~scoped_interpreter() {
        if (is_valid) {
            finalize_interpreter();
        }
    }

private:
    bool is_valid = true;
};

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
