/*
    pybind11/embed.h: Support for embedding the interpreter

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "pybind11.h"
#include "eval.h"

#include <memory>
#include <vector>

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
#define PYBIND11_EMBEDDED_MODULE(name, variable)                                                  \
    static ::pybind11::module_::module_def PYBIND11_CONCAT(pybind11_module_def_, name);           \
    static void PYBIND11_CONCAT(pybind11_init_, name)(::pybind11::module_ &);                     \
    static PyObject PYBIND11_CONCAT(*pybind11_init_wrapper_, name)() {                            \
        auto m = ::pybind11::module_::create_extension_module(                                    \
            PYBIND11_TOSTRING(name), nullptr, &PYBIND11_CONCAT(pybind11_module_def_, name));      \
        try {                                                                                     \
            PYBIND11_CONCAT(pybind11_init_, name)(m);                                             \
            return m.ptr();                                                                       \
        }                                                                                         \
        PYBIND11_CATCH_INIT_EXCEPTIONS                                                            \
    }                                                                                             \
    PYBIND11_EMBEDDED_MODULE_IMPL(name)                                                           \
    ::pybind11::detail::embedded_module PYBIND11_CONCAT(pybind11_module_, name)(                  \
        PYBIND11_TOSTRING(name), PYBIND11_CONCAT(pybind11_init_impl_, name));                     \
    void PYBIND11_CONCAT(pybind11_init_, name)(::pybind11::module_                                \
                                               & variable) // NOLINT(bugprone-macro-parentheses)

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
        if (Py_IsInitialized() != 0)
            pybind11_fail("Can't add new modules after the interpreter has been initialized");

        auto result = PyImport_AppendInittab(name, init);
        if (result == -1)
            pybind11_fail("Insufficient memory to add a new module");
    }
};

struct wide_char_arg_deleter {
    void operator()(wchar_t *ptr) const {
#if PY_VERSION_HEX >= 0x030500f0
        // API docs: https://docs.python.org/3/c-api/sys.html#c.Py_DecodeLocale
        PyMem_RawFree(ptr);
#else
        delete[] ptr;
#endif
    }
};

inline wchar_t *widen_chars(const char *safe_arg) {
#if PY_VERSION_HEX >= 0x030500f0
    wchar_t *widened_arg = Py_DecodeLocale(safe_arg, nullptr);
#else
    wchar_t *widened_arg = nullptr;

// warning C4996: 'mbstowcs': This function or variable may be unsafe.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable:4996)
#endif

#    if defined(HAVE_BROKEN_MBSTOWCS) && HAVE_BROKEN_MBSTOWCS
    size_t count = std::strlen(safe_arg);
#    else
    size_t count = std::mbstowcs(nullptr, safe_arg, 0);
#    endif
    if (count != static_cast<size_t>(-1)) {
        widened_arg = new wchar_t[count + 1];
        mbstowcs(widened_arg, safe_arg, count + 1);
    }

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#endif
    return widened_arg;
}

/// Python 2.x/3.x-compatible version of `PySys_SetArgv`
inline void set_interpreter_argv(int argc, const char *const *argv, bool add_program_dir_to_path) {
    // Before it was special-cased in python 3.8, passing an empty or null argv
    // caused a segfault, so we have to reimplement the special case ourselves.
    bool special_case = (argv == nullptr || argc <= 0);

    const char *const empty_argv[]{"\0"};
    const char *const *safe_argv = special_case ? empty_argv : argv;
    if (special_case)
        argc = 1;

    auto argv_size = static_cast<size_t>(argc);
#if PY_MAJOR_VERSION >= 3
    // SetArgv* on python 3 takes wchar_t, so we have to convert.
    std::unique_ptr<wchar_t *[]> widened_argv(new wchar_t *[argv_size]);
    std::vector<std::unique_ptr<wchar_t[], wide_char_arg_deleter>> widened_argv_entries;
    widened_argv_entries.reserve(argv_size);
    for (size_t ii = 0; ii < argv_size; ++ii) {
        widened_argv_entries.emplace_back(widen_chars(safe_argv[ii]));
        if (!widened_argv_entries.back()) {
            // A null here indicates a character-encoding failure or the python
            // interpreter out of memory. Give up.
            return;
        }
        widened_argv[ii] = widened_argv_entries.back().get();
    }

    auto pysys_argv = widened_argv.get();
#else
    // python 2.x
    std::vector<std::string> strings{safe_argv, safe_argv + argv_size};
    std::vector<char *> char_strings{argv_size};
    for (std::size_t i = 0; i < argv_size; ++i)
        char_strings[i] = &strings[i][0];
    char **pysys_argv = char_strings.data();
#endif

    PySys_SetArgvEx(argc, pysys_argv, static_cast<int>(add_program_dir_to_path));
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

    The remaining optional parameters, `argc`, `argv`, and `add_program_dir_to_path` are
    used to populate ``sys.argv`` and ``sys.path``.
    See the |PySys_SetArgvEx documentation|_ for details.

    .. _Python documentation: https://docs.python.org/3/c-api/init.html#c.Py_InitializeEx
    .. |PySys_SetArgvEx documentation| replace:: ``PySys_SetArgvEx`` documentation
    .. _PySys_SetArgvEx documentation: https://docs.python.org/3/c-api/init.html#c.PySys_SetArgvEx
 \endrst */
inline void initialize_interpreter(bool init_signal_handlers = true,
                                   int argc = 0,
                                   const char *const *argv = nullptr,
                                   bool add_program_dir_to_path = true) {
    if (Py_IsInitialized() != 0)
        pybind11_fail("The interpreter is already running");

    Py_InitializeEx(init_signal_handlers ? 1 : 0);

    detail::set_interpreter_argv(argc, argv, add_program_dir_to_path);
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
    explicit scoped_interpreter(bool init_signal_handlers = true,
                                int argc = 0,
                                const char *const *argv = nullptr,
                                bool add_program_dir_to_path = true) {
        initialize_interpreter(init_signal_handlers, argc, argv, add_program_dir_to_path);
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
