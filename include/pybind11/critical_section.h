// Copyright (c) 2016-2025 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include "pytypes.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

/// This does not do anything if there's a GIL. On free-threaded Python,
/// it locks an object. This uses the CPython API, which has limits
class scoped_critical_section {
public:
#ifdef Py_GIL_DISABLED
    explicit scoped_critical_section(handle obj) : has2(false) {
        PyCriticalSection_Begin(&section, obj.ptr());
    }

    scoped_critical_section(handle obj1, handle obj2) : has2(true) {
        PyCriticalSection2_Begin(&section2, obj1.ptr(), obj2.ptr());
    }

    ~scoped_critical_section() {
        if (has2) {
            PyCriticalSection2_End(&section2);
        } else {
            PyCriticalSection_End(&section);
        }
    }
#else
    explicit scoped_critical_section(handle) {};
    scoped_critical_section(handle, handle) {};
    ~scoped_critical_section() = default;
#endif

    scoped_critical_section(const scoped_critical_section &) = delete;
    scoped_critical_section &operator=(const scoped_critical_section &) = delete;

private:
#ifdef Py_GIL_DISABLED
    bool has2;
    union {
        PyCriticalSection section;
        PyCriticalSection2 section2;
    };
#endif
};

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
