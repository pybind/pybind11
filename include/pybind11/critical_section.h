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
    scoped_critical_section(handle obj1, handle obj2) : m_ptr1(obj1.ptr()), m_ptr2(obj2.ptr()) {
        if (m_ptr1 == nullptr) {
            std::swap(m_ptr1, m_ptr2);
        }
        if (m_ptr2 != nullptr) {
            PyCriticalSection2_Begin(&section2, m_ptr1, m_ptr2);
        } else if (m_ptr1 != nullptr) {
            PyCriticalSection_Begin(&section, m_ptr1);
        }
    }

    explicit scoped_critical_section(handle obj) : m_ptr1(obj.ptr()) {
        if (m_ptr1 != nullptr) {
            PyCriticalSection_Begin(&section, m_ptr1);
        }
    }

    ~scoped_critical_section() {
        if (m_ptr2 != nullptr) {
            PyCriticalSection2_End(&section2);
        } else if (m_ptr1 != nullptr) {
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
    PyObject *m_ptr1{nullptr};
    PyObject *m_ptr2{nullptr};
    union {
        PyCriticalSection section;
        PyCriticalSection2 section2;
    };
#endif
};

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
