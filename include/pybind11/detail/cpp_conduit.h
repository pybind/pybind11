// Copyright (c) 2024 The pybind Community.

#pragma once

#include <pybind11/pytypes.h>

#include "common.h"
#include "internals.h"

#include <typeinfo>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

bool type_is_managed_by_our_internals(PyTypeObject *type_obj);

bool is_instance_method_of_type(PyTypeObject *type_obj, PyObject *attr_name);

object try_get_cpp_conduit_method(PyObject *obj);

void *try_raw_pointer_ephemeral_from_cpp_conduit(handle src, const std::type_info *cpp_type_info);

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

#ifndef PYBIND11_PRECOMPILED
#    include "cpp_conduit-inl.h" // IWYU pragma: export
#endif
