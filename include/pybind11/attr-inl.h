/*
    pybind11/attr-inl.h: Out-of-line definitions for attr.h

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

// Every function defined here must start with PYBIND11_INLINE (or
// PYBIND11_NOINLINE_ATTR PYBIND11_INLINE). In the default header-only mode this file is
// included at the bottom of attr.h; when PYBIND11_PRECOMPILED is defined it is only
// compiled into the pybind11 static library (see src/).

#pragma once

#include "attr.h"

#include <string>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

PYBIND11_NOINLINE_ATTR PYBIND11_INLINE type_record::type_record()
    : multiple_inheritance(false), dynamic_attr(false), buffer_protocol(false),
      module_local(false), is_final(false), release_gil_before_calling_cpp_dtor(false) {}

PYBIND11_NOINLINE_ATTR PYBIND11_INLINE void type_record::add_base(const std::type_info &base,
                                                                  void *(*caster)(void *) ) {
    auto *base_info = detail::get_type_info(base, false);
    if (!base_info) {
        std::string tname(base.name());
        detail::clean_type_id(tname);
        pybind11_fail("generic_type: type \"" + std::string(name)
                      + "\" referenced unknown base type \"" + tname + "\"");
    }

    // SMART_HOLDER_BAKEIN_FOLLOW_ON: Refine holder compatibility checks.
    bool this_has_unique_ptr_holder = (holder_enum_v == holder_enum_t::std_unique_ptr);
    bool base_has_unique_ptr_holder = (base_info->holder_enum_v == holder_enum_t::std_unique_ptr);
    if (this_has_unique_ptr_holder != base_has_unique_ptr_holder) {
        std::string tname(base.name());
        detail::clean_type_id(tname);
        pybind11_fail("generic_type: type \"" + std::string(name) + "\" "
                      + (this_has_unique_ptr_holder ? "does not have" : "has")
                      + " a non-default holder type while its base \"" + tname + "\" "
                      + (base_has_unique_ptr_holder ? "does not" : "does"));
    }

    bases.append(reinterpret_cast<PyObject *>(base_info->type));

#ifdef PYBIND11_BACKWARD_COMPATIBILITY_TP_DICTOFFSET
    dynamic_attr |= base_info->type->tp_dictoffset != 0;
#else
    dynamic_attr |= (PyType_GetFlags(base_info->type) & Py_TPFLAGS_MANAGED_DICT) != 0;
#endif

    if (caster) {
        base_info->implicit_casts.emplace_back(type, caster);
    }
}

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
