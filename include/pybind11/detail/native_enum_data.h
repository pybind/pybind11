// Copyright (c) 2022-2025 The pybind Community.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#define PYBIND11_HAS_NATIVE_ENUM

#include "../pytypes.h"
#include "common.h"
#include "internals.h"

#include <string>
#include <typeindex>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

class native_enum_data {
public:
    native_enum_data(const object &parent_scope,
                     const char *enum_name,
                     const std::type_index &enum_type_index,
                     bool use_int_enum)
        : parent_scope(parent_scope), enum_name_encoded{enum_name},
          enum_type_index{enum_type_index}, use_int_enum{use_int_enum}, enum_name{enum_name} {}

    void finalize();

    native_enum_data(const native_enum_data &) = delete;
    native_enum_data &operator=(const native_enum_data &) = delete;

    void disarm_correct_use_check() const { correct_use_check = false; }
    void arm_correct_use_check() const { correct_use_check = true; }

    // This is a separate public function only to enable easy unit testing.
    std::string was_not_added_error_message() const {
        return "`native_enum` was not added to any module."
               " Use e.g. `m += native_enum<...>(\""
               + enum_name_encoded + "\", ...)` to fix.";
    }

#if !defined(NDEBUG)
    // This dtor cannot easily be unit tested because it terminates the process.
    ~native_enum_data() {
        if (correct_use_check) {
            pybind11_fail(was_not_added_error_message());
        }
    }
#endif

private:
    mutable bool correct_use_check{false};

public:
    object parent_scope;
    std::string enum_name_encoded;
    std::type_index enum_type_index;
    bool use_int_enum;
    bool export_values_flag{false};
    str enum_name;
    list members;
    list docs;
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
    disarm_correct_use_check();
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
