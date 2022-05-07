// Copyright (c) 2022 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

// #########################################################################
// PLEASE UPDATE docs/advanced/cast/custom.rst IF ANY CHANGES ARE MADE HERE.
// #########################################################################

#include "pybind11_tests.h"

#include <string>

// 0: Use the older pybind11::detail::type_caster<T> specialization mechanism.
// 1: Use pybind11_select_caster declaration in user_space.
// 2: Use pybind11_select_caster friend function declaration.
#ifndef PYBIND11_TEST_DOCS_ADVANCED_CAST_CUSTOM_ALTERNATIVE
#    define PYBIND11_TEST_DOCS_ADVANCED_CAST_CUSTOM_ALTERNATIVE 1
#endif

namespace user_space {

#if PYBIND11_TEST_DOCS_ADVANCED_CAST_CUSTOM_ALTERNATIVE == 2
struct inty_type_caster;
#endif

struct inty {
    long long_value;
#if PYBIND11_TEST_DOCS_ADVANCED_CAST_CUSTOM_ALTERNATIVE == 2
    friend inty_type_caster pybind11_select_caster(inty *);
#endif
};

std::string to_string(const inty &s) { return std::to_string(s.long_value); }

inty return_42() { return inty{42}; }

} // namespace user_space

namespace user_space {

struct inty_type_caster {
public:
    // This macro establishes the name 'inty' in function signatures and declares a local variable
    // 'value' of type inty.
    PYBIND11_TYPE_CASTER(inty, pybind11::detail::const_name("inty"));

    // Python -> C++: convert a PyObject into an inty instance or return false upon failure. The
    // second argument indicates whether implicit conversions should be allowed.
    bool load(pybind11::handle src, bool) {
        // Extract PyObject from handle.
        PyObject *source = src.ptr();
        // Try converting into a Python integer value.
        PyObject *tmp = PyNumber_Long(source);
        if (!tmp) {
            return false;
        }
        // Now try to convert into a C++ int.
        value.long_value = PyLong_AsLong(tmp);
        Py_DECREF(tmp);
        // Ensure PyLong_AsLong succeeded (to catch out-of-range errors etc).
        if (PyErr_Occurred()) {
            PyErr_Clear();
            return false;
        }
        return true;
    }
    // C++ -> Python: convert an inty instance into a Python object. The second and third arguments
    // are used to indicate the return value policy and parent object (for
    // return_value_policy::reference_internal) and are often ignored by custom casters.
    static pybind11::handle
    cast(inty src, pybind11::return_value_policy /* policy */, pybind11::handle /* parent */) {
        return PyLong_FromLong(src.long_value);
    }
};

} // namespace user_space

#if PYBIND11_TEST_DOCS_ADVANCED_CAST_CUSTOM_ALTERNATIVE == 1
namespace user_space {

inty_type_caster pybind11_select_caster(inty *);

} // namespace user_space
#endif

#if PYBIND11_TEST_DOCS_ADVANCED_CAST_CUSTOM_ALTERNATIVE == 0
namespace pybind11 {
namespace detail {
template <>
struct type_caster<user_space::inty> : user_space::inty_type_caster {};
} // namespace detail
} // namespace pybind11
#endif

TEST_SUBMODULE(docs_advanced_cast_custom, m) {
    m.def("to_string", user_space::to_string);
    m.def("return_42", user_space::return_42);
}
