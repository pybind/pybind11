#include "embed.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

PYBIND11_NAMESPACE_BEGIN(detail)

PYBIND11_INLINE embedded_module::embedded_module(const char *name, init_t init) {
    if (Py_IsInitialized())
        pybind11_fail("Can't add new modules after the interpreter has been initialized");

    auto result = PyImport_AppendInittab(name, init);
    if (result == -1)
        pybind11_fail("Insufficient memory to add a new module");
}

PYBIND11_NAMESPACE_END(detail)

PYBIND11_INLINE void initialize_interpreter(bool init_signal_handlers) {
    if (Py_IsInitialized())
        pybind11_fail("The interpreter is already running");

    Py_InitializeEx(init_signal_handlers ? 1 : 0);

    // Make .py files in the working directory available by default
    module::import("sys").attr("path").cast<list>().append(".");
}

PYBIND11_INLINE void finalize_interpreter() {
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

PYBIND11_INLINE scoped_interpreter::scoped_interpreter(bool init_signal_handlers) {
    initialize_interpreter(init_signal_handlers);
}

PYBIND11_INLINE scoped_interpreter::~scoped_interpreter() {
    if (is_valid)
        finalize_interpreter();
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
