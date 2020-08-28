#include "options.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

PYBIND11_INLINE options::options() : previous_state(global_state()) {}

PYBIND11_INLINE options::~options() {
    global_state() = previous_state;
}

PYBIND11_INLINE options& options::disable_user_defined_docstrings() & { global_state().show_user_defined_docstrings = false; return *this; }

PYBIND11_INLINE options& options::enable_user_defined_docstrings() & { global_state().show_user_defined_docstrings = true; return *this; }

PYBIND11_INLINE options& options::disable_function_signatures() & { global_state().show_function_signatures = false; return *this; }

PYBIND11_INLINE options& options::enable_function_signatures() & { global_state().show_function_signatures = true; return *this; }

PYBIND11_INLINE bool options::show_user_defined_docstrings() { return global_state().show_user_defined_docstrings; }

PYBIND11_INLINE bool options::show_function_signatures() { return global_state().show_function_signatures; }

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
