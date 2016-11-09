/*
    pybind11/docstring_options.h: selectively suppress the automatic generation of docstrings and function signatures.

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"

NAMESPACE_BEGIN(pybind11)

class docstring_options {
public:

    // Default RAII constructor, which leaves current settings as they are.
    docstring_options() {
        previous_state = global_state();
    }

    // Class is non-copyable.
    docstring_options(const docstring_options&) = delete;
    docstring_options& operator=(const docstring_options&) = delete;

    // Destructor, which restores settings that were in effect before.
    ~docstring_options() {
        global_state() = previous_state;
    }

    // Setter methods (affect the global state):

    docstring_options& disable_user_defined() & { global_state().show_user_defined = false; return *this; }

    docstring_options& enable_user_defined() & { global_state().show_user_defined = true; return *this; }

    docstring_options& disable_signatures() & { global_state().show_signatures = false; return *this; }

    docstring_options& enable_signatures() & { global_state().show_signatures = true; return *this; }

    // Getter methods (return the global state):

    static bool show_user_defined() { return global_state().show_user_defined; }

    static bool show_signatures() { return global_state().show_signatures; }

    // This type is not meant to be allocated on the heap.
    void* operator new(size_t) = delete;

private:

    // A collection of flags that control the generation of docstrings by pybind11.
    struct state {
        bool show_user_defined; //< Include user-defined texts in docstrings.
        bool show_signatures;   //< Include auto-generated function signatures in docstrings.
    };

    static state &global_state() {
        static state instance = { true, true };
        return instance;
    }

    state previous_state;
};

NAMESPACE_END(pybind11)
