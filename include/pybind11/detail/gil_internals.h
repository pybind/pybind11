/*
    pybind11/detail/gil_internals.h: Containers for GIL implementations

    Copyright (c) 2018 Kitware Inc. <kyle.edwards@kitware.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"
#include "internals.h"

#include "../gil.h"

NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

class basic_gil_impl;

// Change this to change the default GIL implementation
typedef basic_gil_impl default_gil_impl;

NAMESPACE_BEGIN(detail)

#define PYBIND11_GIL_INTERNALS_MAJOR_VERSION    1
#define PYBIND11_GIL_INTERNALS_MINOR_VERSION    0
#define PYBIND11_GIL_INTERNALS_ID               "__pybind11_gil_internals__"

struct gil_container_base {
    virtual ~gil_container_base() = default;
};

template<typename Obj>
struct gil_container : public gil_container_base {
    Obj obj;
};

struct gil_impl_container_base {
    virtual gil_container_base *create_release() const = 0;
    virtual gil_container_base *create_acquire() const = 0;
    //virtual bool thread_has_impl() const = 0;
};

template<typename Impl>
struct gil_impl_container : public gil_impl_container_base {
    gil_container_base *create_release() const override {
        return new gil_container<typename Impl::release>;
    }

    gil_container_base *create_acquire() const override {
        return new gil_container<typename Impl::acquire>;
    }

    /*bool thread_has_impl() const override {
        return Impl::thread_has_impl();
    }*/
};

/// Internal data structure used to track desired GIL behavior.
struct gil_internals {
    unsigned int major_version;
    unsigned int minor_version;
};

/// V1 of the gil_internals structure.
struct gil_internals_v1_0 : public gil_internals {
    gil_impl_container<default_gil_impl> default_impl;
    std::unique_ptr<gil_impl_container_base> selected_impl;
};

typedef gil_internals_v1_0 gil_internals_current;

inline gil_internals **&get_gil_internals_pp() {
    static gil_internals **gil_internals_pp = nullptr;
    return gil_internals_pp;
}

/// Return a reference to the current `gil_internals` data
PYBIND11_NOINLINE inline gil_internals &get_gil_internals_base() {
    auto **&gil_internals_pp = get_gil_internals_pp();
    if (gil_internals_pp && *gil_internals_pp)
        return **gil_internals_pp;

    constexpr auto *id = PYBIND11_GIL_INTERNALS_ID;
    auto builtins = handle(PyEval_GetBuiltins());
    if (builtins.contains(id) && isinstance<capsule>(builtins[id])) {
        gil_internals_pp = static_cast<gil_internals **>(capsule(builtins[id]));
    } else {
        if (!gil_internals_pp) gil_internals_pp = new gil_internals*();
        gil_internals *&gil_internals_ptr = *gil_internals_pp;
        gil_internals_ptr = new gil_internals_current;
        builtins[id] = capsule(gil_internals_pp);

        gil_internals_v1_0 *ptr = static_cast<gil_internals_current *>(gil_internals_ptr);
        ptr->major_version = PYBIND11_GIL_INTERNALS_MAJOR_VERSION;
        ptr->minor_version = PYBIND11_GIL_INTERNALS_MINOR_VERSION;
        ptr->selected_impl = nullptr;
    }
    return **gil_internals_pp;
}

// When PYBIND11_GIL_INTERNALS_MINOR_VERSION is 0, a warning is triggered
#pragma GCC diagnostic ignored "-Wtype-limits"
PYBIND11_NOINLINE inline gil_internals_current &get_gil_internals() {
    gil_internals &internals = get_gil_internals_base();
    if (internals.major_version == PYBIND11_GIL_INTERNALS_MAJOR_VERSION &&
        internals.minor_version >= PYBIND11_GIL_INTERNALS_MINOR_VERSION) {
        return static_cast<gil_internals_current &>(internals);
    } else {
        throw std::runtime_error("Incompatible gil_internals version");
    }
}

template<typename Impl>
void select_gil_impl() {
    gil_internals_current &internals = get_gil_internals();

    if (internals.selected_impl) {
        if (!same_type(typeid(*internals.selected_impl), typeid(gil_impl_container<Impl>)))
            throw std::runtime_error("Conflicting GIL requirements");
    } else {
        internals.selected_impl.reset(new gil_impl_container<Impl>);
    }
}

PYBIND11_NOINLINE inline gil_impl_container_base &get_selected_gil_impl() {
    gil_internals_current &internals = get_gil_internals();

    if (internals.selected_impl) {
        return *internals.selected_impl;
    } else {
        return internals.default_impl;
    }
}

NAMESPACE_END(detail)
NAMESPACE_END(PYBIND11_NAMESPACE)
