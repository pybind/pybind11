/*
    pybind11/docstring_options.h: selectively suppress the automatic generation of docstrings and function signatures.

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"

NAMESPACE_BEGIN(pybind11)

class docstring_options
{
  public:

    struct state {
        bool show_user_defined; //< Include user-defined texts in docstrings.
        bool show_signatures;   //< Inclcude auto-generated function signatures in docstrings.
    };

    PYBIND11_NOINLINE inline static state &global_state() {
        static state instance = { true, true };
        return instance;
    }

    docstring_options(bool show_all)
    {
        previous_state = global_state();
        global_state().show_user_defined = show_all;
        global_state().show_signatures = show_all;
    }

    docstring_options(bool show_user_defined, bool show_signatures)
    {
        previous_state = global_state();
        global_state().show_user_defined = show_user_defined;
        global_state().show_signatures = show_signatures;
    }

    ~docstring_options()
    {
        global_state() = previous_state;
    }

    void disable_user_defined() { global_state().show_user_defined = false; }

    void enable_user_defined() { global_state().show_user_defined = true; }

    void disable_signatures() { global_state().show_signatures = false; }

    void enable_signatures() { global_state().show_signatures = true; }

    void disable_all() {
        global_state().show_user_defined = false;
        global_state().show_signatures = false;
    }

    void enable_all() {
        global_state().show_user_defined = true;
        global_state().show_signatures = true;
    }

  private:

    // For saving the flags on the stack:
    state previous_state;
};

NAMESPACE_END(pybind11)
