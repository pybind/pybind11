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

    // Default constructor, which leaves current settings as they are.
    docstring_options() {
        previous_state = global_state();
    }

    // Initializing constructor, which overrides current global state.
    docstring_options(bool show_user_defined, bool show_signatures) {
        previous_state = global_state();
        global_state().show_user_defined = show_user_defined;
        global_state().show_signatures = show_signatures;
    }

    // Class is non-copyable.
    docstring_options(const docstring_options&) = delete;
    docstring_options& operator=(const docstring_options&) = delete;

    // Destructor, which restores settings that were in effect before.
    ~docstring_options() {
        global_state() = previous_state;
    }

    // Setter methods (affect the global state):

    void disable_user_defined() { global_state().show_user_defined = false; }

    void enable_user_defined() { global_state().show_user_defined = true; }

    void disable_signatures() { global_state().show_signatures = false; }

    void enable_signatures() { global_state().show_signatures = true; }

    // Getter methods (return the global state):

    static bool show_user_defined() { return global_state().show_user_defined; }

    static bool show_signatures() { return global_state().show_signatures; }

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
