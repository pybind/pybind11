// Copyright (c) 2022-2025 The pybind Community.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#define PYBIND11_HAS_NATIVE_ENUM

#include "../pytypes.h"
#include "common.h"
#include "internals.h"

#include <cassert>
#include <string>
#include <typeindex>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

// This is a separate function only to enable easy unit testing.
inline std::string
native_enum_missing_finalize_error_message(const std::string &enum_name_encoded) {
    return "pybind11::native_enum<...>(\"" + enum_name_encoded + "\", ...): MISSING .finalize()";
}

class native_enum_data {
public:
    native_enum_data(const object &parent_scope,
                     const char *enum_name,
                     const std::type_index &enum_type_index,
                     bool use_int_enum)
        : enum_name_encoded{enum_name}, enum_type_index{enum_type_index}, enum_name{enum_name},
          parent_scope(parent_scope), export_values_flag{false}, use_int_enum{use_int_enum},
          finalize_needed{false} {}

    void finalize();

    native_enum_data(const native_enum_data &) = delete;
    native_enum_data &operator=(const native_enum_data &) = delete;

#if !defined(NDEBUG)
    // This dtor cannot easily be unit tested because it terminates the process.
    ~native_enum_data() {
        if (finalize_needed) {
            pybind11_fail(native_enum_missing_finalize_error_message(enum_name_encoded));
        }
    }
#endif

protected:
    void disarm_finalize_check(const char *error_context) {
        if (!finalize_needed) {
            pybind11_fail("pybind11::native_enum<...>(\"" + enum_name_encoded
                          + "\"): " + error_context);
        }
        finalize_needed = false;
    }

    void arm_finalize_check() {
        assert(!finalize_needed);
        finalize_needed = true;
    }

    std::string enum_name_encoded;
    std::type_index enum_type_index;

private:
    str enum_name;
    object parent_scope;

protected:
    list members;
    list docs;
    bool export_values_flag : 1;

private:
    bool use_int_enum : 1;
    bool finalize_needed : 1;
};

inline void global_internals_native_enum_type_map_set_item(const std::type_index &enum_type_index,
                                                           PyObject *py_enum) {
    with_internals(
        [&](internals &internals) { internals.native_enum_type_map[enum_type_index] = py_enum; });
}

inline handle
global_internals_native_enum_type_map_get_item(const std::type_index &enum_type_index) {
    return with_internals([&](internals &internals) {
        auto found = internals.native_enum_type_map.find(enum_type_index);
        if (found != internals.native_enum_type_map.end()) {
            return handle(found->second);
        }
        return handle();
    });
}

inline bool
global_internals_native_enum_type_map_contains(const std::type_index &enum_type_index) {
    return with_internals([&](internals &internals) {
        return internals.native_enum_type_map.count(enum_type_index) != 0;
    });
}

inline void native_enum_data::finalize() {
    disarm_finalize_check("DOUBLE finalize");
    if (hasattr(parent_scope, enum_name)) {
        pybind11_fail("pybind11::native_enum<...>(\"" + enum_name_encoded
                      + "\"): an object with that name is already defined");
    }
    auto enum_module = reinterpret_steal<object>(PyImport_ImportModule("enum"));
    if (!enum_module) {
        raise_from(PyExc_SystemError,
                   "`import enum` FAILED at " __FILE__ ":" PYBIND11_TOSTRING(__LINE__));
        throw error_already_set();
    }
    auto py_enum_type = enum_module.attr(use_int_enum ? "IntEnum" : "Enum");
    auto py_enum = py_enum_type(enum_name, members);
    object module_name = get_module_name_if_available(parent_scope);
    if (module_name) {
        py_enum.attr("__module__") = module_name;
    }
    parent_scope.attr(enum_name) = py_enum;
    if (export_values_flag) {
        for (auto member : members) {
            auto member_name = member[int_(0)];
            if (hasattr(parent_scope, member_name)) {
                pybind11_fail("pybind11::native_enum<...>(\"" + enum_name_encoded + "\").value(\""
                              + member_name.cast<std::string>()
                              + "\"): an object with that name is already defined");
            }
            parent_scope.attr(member_name) = py_enum[member_name];
        }
    }
    for (auto doc : docs) {
        py_enum[doc[int_(0)]].attr("__doc__") = doc[int_(1)];
    }
    global_internals_native_enum_type_map_set_item(enum_type_index, py_enum.release().ptr());
}

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
