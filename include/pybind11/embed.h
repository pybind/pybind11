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
#  error Embedding the interpreter is not supported with PyPy
#endif

#if PY_MAJOR_VERSION >= 3
#  define PYBIND11_EMBEDDED_MODULE_IMPL(name)            \
      extern "C" PyObject *pybind11_init_impl_##name();  \
      extern "C" PyObject *pybind11_init_impl_##name() { \
          return pybind11_init_wrapper_##name();         \
      }
#else
#  define PYBIND11_EMBEDDED_MODULE_IMPL(name)            \
      extern "C" void pybind11_init_impl_##name();       \
      extern "C" void pybind11_init_impl_##name() {      \
          pybind11_init_wrapper_##name();                \
      }
#endif

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
 \endrst */
#define PYBIND11_EMBEDDED_MODULE(name, variable)                              \
    static void PYBIND11_CONCAT(pybind11_init_, name)(pybind11::module &);    \
    static PyObject PYBIND11_CONCAT(*pybind11_init_wrapper_, name)() {        \
        auto m = pybind11::module(PYBIND11_TOSTRING(name));                   \
        try {                                                                 \
            PYBIND11_CONCAT(pybind11_init_, name)(m);                         \
            return m.ptr();                                                   \
        } catch (pybind11::error_already_set &e) {                            \
            PyErr_SetString(PyExc_ImportError, e.what());                     \
            return nullptr;                                                   \
        } catch (const std::exception &e) {                                   \
            PyErr_SetString(PyExc_ImportError, e.what());                     \
            return nullptr;                                                   \
        }                                                                     \
    }                                                                         \
    PYBIND11_EMBEDDED_MODULE_IMPL(name)                                       \
    pybind11::detail::embedded_module PYBIND11_CONCAT(pybind11_module_, name) \
                              (PYBIND11_TOSTRING(name),             \
                               PYBIND11_CONCAT(pybind11_init_impl_, name));   \
    void PYBIND11_CONCAT(pybind11_init_, name)(pybind11::module &variable)


PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

/// Python 2.7/3.x compatible version of `PyImport_AppendInittab` and error checks.
struct embedded_module {
#if PY_MAJOR_VERSION >= 3
    using init_t = PyObject *(*)();
#else
    using init_t = void (*)();
#endif
    embedded_module(const char *name, init_t init) {
        if (Py_IsInitialized())
            pybind11_fail("Can't add new modules after the interpreter has been initialized");

        auto result = PyImport_AppendInittab(name, init);
        if (result == -1)
            pybind11_fail("Insufficient memory to add a new module");
    }
};

/// Python 2.x/3.x-compatible version of `PySys_SetArgv`
inline void set_interpreter_argv(int argc, char** argv, bool add_current_dir_to_path) {
    // Before it was special-cased in python 3.8, passing an empty or null argv
    // caused a segfault, so we have to reimplement the special case ourselves.
    char** safe_argv = argv;
    if (nullptr == argv || argc <= 0) {
        safe_argv = new char*[1];
        if (nullptr == safe_argv) return;
        safe_argv[0] = new char[1];
        if (nullptr == safe_argv[0]) {
            delete[] safe_argv;
            return;
        }
        safe_argv[0][0] = '\0';
        argc = 1;
    }
#if PY_MAJOR_VERSION >= 3
    // SetArgv* on python 3 takes wchar_t, so we have to convert.
    wchar_t** widened_argv = new wchar_t*[static_cast<unsigned>(argc)];
    for (int ii = 0; ii < argc; ++ii) {
#  if PY_MINOR_VERSION >= 5
        // From Python 3.5 onwards, we're supposed to use Py_DecodeLocale to
        // generate the wchar_t version of argv.
        widened_argv[ii] = Py_DecodeLocale(safe_argv[ii], nullptr);
#    define FREE_WIDENED_ARG(X) PyMem_RawFree(X)
#  else
        // Before Python 3.5, we're stuck with mbstowcs, which may or may not
        // actually work. Mercifully, pyconfig.h provides this define:
#    ifdef HAVE_BROKEN_MBSTOWCS
        size_t count = strlen(safe_argv[ii]);
#    else
        size_t count = mbstowcs(nullptr, safe_argv[ii], 0);
#    endif
        widened_argv[ii] = nullptr;
        if (count != static_cast<size_t>(-1)) {
            widened_argv[ii] = new wchar_t[count + 1];
            mbstowcs(widened_argv[ii], safe_argv[ii], count + 1);
        }
#    define FREE_WIDENED_ARG(X) delete[] X
#  endif
        if (nullptr == widened_argv[ii]) {
            // Either we ran out of memory or had a unicode encoding issue.
            // Free what we've encoded so far and bail.
            for (--ii; ii >= 0; --ii)
                FREE_WIDENED_ARG(widened_argv[ii]);
            return;
        }
    }

#  if PY_MINOR_VERSION < 1 || (PY_MINOR_VERSION == 1 && PY_MICRO_VERSION < 3)
#    define NEED_PYRUN_TO_SANITIZE_PATH 1
    // don't have SetArgvEx yet
    PySys_SetArgv(argc, widened_argv);
#  else
    PySys_SetArgvEx(argc, widened_argv, add_current_dir_to_path ? 1 : 0);
#  endif

    // PySys_SetArgv makes new PyUnicode objects so we can clean up this memory
    if (nullptr != widened_argv) {
        for (int ii = 0; ii < argc; ++ii)
            if (nullptr != widened_argv[ii])
                FREE_WIDENED_ARG(widened_argv[ii]);
        delete[] widened_argv;
    }
#  undef FREE_WIDENED_ARG
#else
    // python 2.x
#  if PY_MINOR_VERSION < 6 || (PY_MINOR_VERSION == 6 && PY_MICRO_VERSION < 6)
#    define NEED_PYRUN_TO_SANITIZE_PATH 1
    // don't have SetArgvEx yet
    PySys_SetArgv(argc, safe_argv);
#  else
    PySys_SetArgvEx(argc, safe_argv, add_current_dir_to_path ? 1 : 0);
#  endif
#endif

#ifdef NEED_PYRUN_TO_SANITIZE_PATH
#  undef NEED_PYRUN_TO_SANITIZE_PATH
    if (!add_current_dir_to_path)
        PyRun_SimpleString("import sys; sys.path.pop(0)\n");
#endif

    // if we allocated new memory to make safe_argv, we need to free it
    if (safe_argv != argv) {
        delete[] safe_argv[0];
        delete[] safe_argv;
    }
}

PYBIND11_NAMESPACE_END(detail)

/** \rst
    Initialize the Python interpreter. No other pybind11 or CPython API functions can be
    called before this is done; with the exception of `PYBIND11_EMBEDDED_MODULE`. The
    optional `init_signal_handlers` parameter can be used to skip the registration of
    signal handlers (see the `Python documentation`_ for details). Calling this function
    again after the interpreter has already been initialized is a fatal error.

    If initializing the Python interpreter fails, then the program is terminated.  (This
    is controlled by the CPython runtime and is an exception to pybind11's normal behavior
    of throwing exceptions on errors.)

    The remaining optional parameters, `argc`, `argv`, and `add_current_dir_to_path` are
    used to populate ``sys.argv`` and ``sys.path``.
    See the |PySys_SetArgvEx documentation|_ for details.

    .. _Python documentation: https://docs.python.org/3/c-api/init.html#c.Py_InitializeEx
    .. |PySys_SetArgvEx documentation| replace:: ``PySys_SetArgvEx`` documentation
    .. _PySys_SetArgvEx documentation: https://docs.python.org/3/c-api/init.html#c.PySys_SetArgvEx
 \endrst */
inline void initialize_interpreter(bool init_signal_handlers = true,
                                   int argc = 0,
                                   char** argv = nullptr,
                                   bool add_current_dir_to_path = true) {
    if (Py_IsInitialized())
        pybind11_fail("The interpreter is already running");

    Py_InitializeEx(init_signal_handlers ? 1 : 0);

    detail::set_interpreter_argv(argc, argv, add_current_dir_to_path);
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
    handle builtins(PyEval_GetBuiltins());
    const char *id = PYBIND11_INTERNALS_ID;

    // Get the internals pointer (without creating it if it doesn't exist).  It's possible for the
    // internals to be created during Py_Finalize() (e.g. if a py::capsule calls `get_internals()`
    // during destruction), so we get the pointer-pointer here and check it after Py_Finalize().
    detail::internals **internals_ptr_ptr = detail::get_internals_pp();
    // It could also be stashed in builtins, so look there too:
    if (builtins.contains(id) && isinstance<capsule>(builtins[id]))
        internals_ptr_ptr = capsule(builtins[id]);

    Py_Finalize();

    if (internals_ptr_ptr) {
        delete *internals_ptr_ptr;
        *internals_ptr_ptr = nullptr;
    }
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
    scoped_interpreter(bool init_signal_handlers = true,
                       int argc = 0,
                       char** argv = nullptr,
                       bool add_current_dir_to_path = true) {
        initialize_interpreter(init_signal_handlers, argc, argv, add_current_dir_to_path);
    }

    scoped_interpreter(const scoped_interpreter &) = delete;
    scoped_interpreter(scoped_interpreter &&other) noexcept { other.is_valid = false; }
    scoped_interpreter &operator=(const scoped_interpreter &) = delete;
    scoped_interpreter &operator=(scoped_interpreter &&) = delete;

    ~scoped_interpreter() {
        if (is_valid)
            finalize_interpreter();
    }

private:
    bool is_valid = true;
};

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
