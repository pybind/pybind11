/*
    pybind11/detail/holder_caster_foreign_helpers.h: Logic to implement
    set_foreign_holder() in copyable_ and movable_holder_caster.

    Copyright (c) 2025 Hudson River Trading LLC <opensource@hudson-trading.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include <pybind11/gil.h>

#include "common.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

struct holder_caster_foreign_helpers {
    struct py_deleter {
        void operator()(const void *) const noexcept {
            // Don't run the deleter if the interpreter has been shut down
            if (Py_IsInitialized() == 0) {
                return;
            }
            gil_scoped_acquire guard;
            Py_DECREF(o);
        }

        PyObject *o;
    };

    template <typename type>
    static auto set_via_shared_from_this(type *value, std::shared_ptr<type> *holder_out)
        -> decltype(value->shared_from_this(), bool()) {
        // object derives from enable_shared_from_this;
        // try to reuse an existing shared_ptr if one is known
        if (auto existing = try_get_shared_from_this(value)) {
            *holder_out = std::static_pointer_cast<type>(existing);
            return true;
        }
        return false;
    }

    template <typename type>
    static bool set_via_shared_from_this(void *, std::shared_ptr<type> *) {
        return false;
    }

    // We only support loading a holder H<T> for foreign T in two cases:
    // - If H = std::shared_ptr, then we can create a new shared_ptr control
    //   block that owns a reference to the original Python object.
    //   (Or if T inherits enable_shared_from_this, we can use that to
    //   locate the original shared_ptr holder.)
    // - If H is a custom holder with always_construct_holder set to true,
    //   we can create a new holder without needing to locate the existing one.
    // In general there is no way to locate the existing holder for a foreign
    // instance, because pymetabind doesn't provide that capability.

    template <typename type>
    static bool set_foreign_holder(handle src, type *value, std::shared_ptr<type> *holder_out) {
        if (value == nullptr) {
            *holder_out = {};
            return true;
        }
        if (set_via_shared_from_this(value, holder_out)) {
            return true;
        }
        *holder_out = std::shared_ptr<type>(value, py_deleter{src.inc_ref().ptr()});
        return true;
    }

    template <typename type>
    static bool
    set_foreign_holder(handle src, const type *value, std::shared_ptr<const type> *holder_out) {
        std::shared_ptr<type> holder_mut;
        if (set_foreign_holder(src, const_cast<type *>(value), &holder_mut)) {
            *holder_out = holder_mut;
            return true;
        }
        return false;
    }

    template <typename type, typename holder,
              typename = enable_if_t<always_construct_holder<holder>::value>>
    static bool set_foreign_holder(handle /*src*/, type *value, holder *holder_out) {
        if (value == nullptr) {
            *holder_out = {};
            return true;
        }
        *holder_out = holder(value);
        return true;
    }

    template <typename type>
    static bool set_foreign_holder(handle, type *, ...) {
        throw cast_error("Unable to cast foreign type to held instance -- "
                         "only std::shared_ptr<T> or a custom holder with the "
                         "3rd argument of PYBIND11_DECLARE_HOLDER_TYPE() set "
                         "to true are supported in this case");
    }
};

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
