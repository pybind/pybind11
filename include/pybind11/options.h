/*
    pybind11/options.h: global settings that are configurable at runtime.

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"

NAMESPACE_BEGIN(pybind11)

class options {
public:

    // Default RAII constructor, which leaves settings as they currently are.
    options() : previous_state(global_state()) {}

    // Class is non-copyable.
    options(const options&) = delete;
    options& operator=(const options&) = delete;

    // Destructor, which restores settings that were in effect before.
    ~options() {
        global_state() = previous_state;
    }

    // Setter methods (affect the global state):

    options& disable_user_defined_docstrings() & { global_state().show_user_defined_docstrings = false; return *this; }

    options& enable_user_defined_docstrings() & { global_state().show_user_defined_docstrings = true; return *this; }

    options& disable_function_signatures() & { global_state().show_function_signatures = false; return *this; }

    options& enable_function_signatures() & { global_state().show_function_signatures = true; return *this; }

    options& set_type_error_print_repr() { global_state().type_error_print_repr = true; return *this; }
    options& set_type_error_print_str() { global_state().type_error_print_repr = false; return *this; }



    // Getter methods (return the global state):

    static bool show_user_defined_docstrings() { return global_state().show_user_defined_docstrings; }

    static bool show_function_signatures() { return global_state().show_function_signatures; }

    static bool type_error_print_repr() { return global_state().type_error_print_repr; }
    static bool type_error_print_str() { return !global_state().type_error_print_repr; }

    // This type is not meant to be allocated on the heap.
    void* operator new(size_t) = delete;

private:

    struct state {
        bool show_user_defined_docstrings = true;  //< Include user-supplied texts in docstrings.
        bool show_function_signatures = true;      //< Include auto-generated function signatures in docstrings.
        bool type_error_print_repr = true;         //< If true, if args mismatch __repr__ of argument is shown if true, 
                                                   //< else __str__ is used
    };

    static state &global_state() {
        static state instance;
        return instance;
    }

    state previous_state;
};

NAMESPACE_END(pybind11)
