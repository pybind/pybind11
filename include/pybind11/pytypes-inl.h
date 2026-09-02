/*
    pybind11/pytypes-inl.h: Out-of-line definitions for pytypes.h

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

// Every function defined here must start with PYBIND11_INLINE (or
// PYBIND11_NOINLINE_ATTR PYBIND11_INLINE). In the default header-only mode this file is
// included at the bottom of pytypes.h; when PYBIND11_PRECOMPILED is defined it is only
// compiled into the pybind11 static library (see src/).

#pragma once

#include "pytypes.h"

#include <string>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

PYBIND11_INLINE error_fetch_and_normalize::error_fetch_and_normalize(const char *called) {
    PyErr_Fetch(&m_type.ptr(), &m_value.ptr(), &m_trace.ptr());
    if (!m_type) {
        pybind11_fail("Internal error: " + std::string(called)
                      + " called while "
                        "Python error indicator not set.");
    }
    const char *exc_type_name_orig = detail::obj_class_name(m_type.ptr());
    if (exc_type_name_orig == nullptr) {
        pybind11_fail("Internal error: " + std::string(called)
                      + " failed to obtain the name "
                        "of the original active exception type.");
    }
    m_lazy_error_string = exc_type_name_orig;
#if PY_VERSION_HEX >= 0x030C0000
    // The presence of __notes__ is likely due to exception normalization
    // errors, although that is not necessarily true, therefore insert a
    // hint only:
    const int has_notes = PyObject_HasAttrString(m_value.ptr(), "__notes__");
    if (has_notes == 1) {
        m_lazy_error_string += "[WITH __notes__]";
    } else if (has_notes == -1) {
        // Ignore secondary errors when probing for __notes__ to avoid leaking a
        // spurious exception while still reporting the original error.
        PyErr_Clear();
    }
#else
    // PyErr_NormalizeException() may change the exception type if there are cascading
    // failures. This can potentially be extremely confusing.
    PyErr_NormalizeException(&m_type.ptr(), &m_value.ptr(), &m_trace.ptr());
    if (m_type.ptr() == nullptr) {
        pybind11_fail("Internal error: " + std::string(called)
                      + " failed to normalize the "
                        "active exception.");
    }
    const char *exc_type_name_norm = detail::obj_class_name(m_type.ptr());
    if (exc_type_name_norm == nullptr) {
        pybind11_fail("Internal error: " + std::string(called)
                      + " failed to obtain the name "
                        "of the normalized active exception type.");
    }
    if (exc_type_name_norm != m_lazy_error_string) {
        std::string msg = std::string(called)
                          + ": MISMATCH of original and normalized "
                            "active exception types: ";
        msg += "ORIGINAL ";
        msg += m_lazy_error_string;
        msg += " REPLACED BY ";
        msg += exc_type_name_norm;
        msg += ": " + format_value_and_trace();
        pybind11_fail(msg);
    }
#endif
}

PYBIND11_INLINE std::string error_fetch_and_normalize::format_value_and_trace() const {
    std::string result;
    std::string message_error_string;
    if (m_value) {
        auto value_str = reinterpret_steal<object>(PyObject_Str(m_value.ptr()));
        constexpr const char *message_unavailable_exc
            = "<MESSAGE UNAVAILABLE DUE TO ANOTHER EXCEPTION>";
        if (!value_str) {
            message_error_string = detail::error_string();
            result = message_unavailable_exc;
        } else {
            // Not using `value_str.cast<std::string>()`, to not potentially throw a secondary
            // error_already_set that will then result in process termination (#4288).
            auto value_bytes = reinterpret_steal<object>(
                PyUnicode_AsEncodedString(value_str.ptr(), "utf-8", "backslashreplace"));
            if (!value_bytes) {
                message_error_string = detail::error_string();
                result = message_unavailable_exc;
            } else {
                char *buffer = nullptr;
                Py_ssize_t length = 0;
                if (PyBytes_AsStringAndSize(value_bytes.ptr(), &buffer, &length) == -1) {
                    message_error_string = detail::error_string();
                    result = message_unavailable_exc;
                } else {
                    result = std::string(buffer, static_cast<std::size_t>(length));
                }
            }
        }
#if PY_VERSION_HEX >= 0x030B0000
        auto notes = reinterpret_steal<object>(PyObject_GetAttrString(m_value.ptr(), "__notes__"));
        if (!notes) {
            PyErr_Clear(); // No notes is good news.
        } else {
            auto len_notes = PyList_Size(notes.ptr());
            if (len_notes < 0) {
                result += "\nFAILURE obtaining len(__notes__): " + detail::error_string();
            } else {
                result += "\n__notes__ (len=" + std::to_string(len_notes) + "):";
                for (ssize_t i = 0; i < len_notes; i++) {
                    PyObject *note = PyList_GET_ITEM(notes.ptr(), i);
                    auto note_bytes = reinterpret_steal<object>(
                        PyUnicode_AsEncodedString(note, "utf-8", "backslashreplace"));
                    if (!note_bytes) {
                        result += "\nFAILURE obtaining __notes__[" + std::to_string(i)
                                  + "]: " + detail::error_string();
                    } else {
                        char *buffer = nullptr;
                        Py_ssize_t length = 0;
                        if (PyBytes_AsStringAndSize(note_bytes.ptr(), &buffer, &length) == -1) {
                            result += "\nFAILURE formatting __notes__[" + std::to_string(i)
                                      + "]: " + detail::error_string();
                        } else {
                            result += '\n';
                            result += std::string(buffer, static_cast<std::size_t>(length));
                        }
                    }
                }
            }
        }
#endif
    } else {
        result = "<MESSAGE UNAVAILABLE>";
    }
    if (result.empty()) {
        result = "<EMPTY MESSAGE>";
    }

    bool have_trace = false;
    if (m_trace) {
#if !defined(PYPY_VERSION) && !defined(GRAALVM_PYTHON)
        auto *tb = reinterpret_cast<PyTracebackObject *>(m_trace.ptr());

        // Get the deepest trace possible.
        while (tb->tb_next) {
            tb = tb->tb_next;
        }

        PyFrameObject *frame = tb->tb_frame;
        Py_XINCREF(frame);
        result += "\n\nAt:\n";
        while (frame) {
            PyCodeObject *f_code = PyFrame_GetCode(frame);
            int lineno = PyFrame_GetLineNumber(frame);
            result += "  ";
            result += handle(f_code->co_filename).cast<std::string>();
            result += '(';
            result += std::to_string(lineno);
            result += "): ";
            result += handle(f_code->co_name).cast<std::string>();
            result += '\n';
            Py_DECREF(f_code);
            auto *b_frame = PyFrame_GetBack(frame);
            Py_DECREF(frame);
            frame = b_frame;
        }

        have_trace = true;
#endif //! defined(PYPY_VERSION)
    }

    if (!message_error_string.empty()) {
        if (!have_trace) {
            result += '\n';
        }
        result += "\nMESSAGE UNAVAILABLE DUE TO EXCEPTION: " + message_error_string;
    }

    return result;
}

PYBIND11_INLINE std::string const &error_fetch_and_normalize::error_string() const {
    if (!m_lazy_error_string_completed) {
        m_lazy_error_string += ": " + format_value_and_trace();
        m_lazy_error_string_completed = true;
    }
    return m_lazy_error_string;
}

PYBIND11_INLINE void error_fetch_and_normalize::restore() {
    if (m_restore_called) {
        pybind11_fail("Internal error: pybind11::detail::error_fetch_and_normalize::restore() "
                      "called a second time. ORIGINAL ERROR: "
                      + error_string());
    }
    PyErr_Restore(m_type.inc_ref().ptr(), m_value.inc_ref().ptr(), m_trace.inc_ref().ptr());
    m_restore_called = true;
}

PYBIND11_INLINE std::string error_string() {
    return error_fetch_and_normalize("pybind11::detail::error_string").error_string();
}

PYBIND11_NAMESPACE_END(detail)

PYBIND11_INLINE void raise_from(PyObject *type, const char *message) {
    // Based on _PyErr_FormatVFromCause:
    // https://github.com/python/cpython/blob/467ab194fc6189d9f7310c89937c51abeac56839/Python/errors.c#L405
    // See https://github.com/pybind/pybind11/pull/2112 for details.
    PyObject *exc = nullptr, *val = nullptr, *val2 = nullptr, *tb = nullptr;

    assert(PyErr_Occurred());
    PyErr_Fetch(&exc, &val, &tb);
    PyErr_NormalizeException(&exc, &val, &tb);
    if (tb != nullptr) {
        PyException_SetTraceback(val, tb);
        Py_DECREF(tb);
    }
    Py_DECREF(exc);
    assert(!PyErr_Occurred());

    PyErr_SetString(type, message);

    PyErr_Fetch(&exc, &val2, &tb);
    PyErr_NormalizeException(&exc, &val2, &tb);
    Py_INCREF(val);
    PyException_SetCause(val2, val);
    PyException_SetContext(val2, val);
    PyErr_Restore(exc, val2, tb);
}

PYBIND11_INLINE void raise_from(error_already_set &err, PyObject *type, const char *message) {
    err.restore();
    raise_from(type, message);
}

/// @cond DUPLICATE
PYBIND11_INLINE memoryview memoryview::from_buffer(void *ptr,
                                                   ssize_t itemsize,
                                                   const char *format,
                                                   detail::any_container<ssize_t> shape,
                                                   detail::any_container<ssize_t> strides,
                                                   bool readonly) {
    size_t ndim = shape->size();
    if (ndim != strides->size()) {
        pybind11_fail("memoryview: shape length doesn't match strides length");
    }
    ssize_t size = ndim != 0u ? 1 : 0;
    for (size_t i = 0; i < ndim; ++i) {
        size *= (*shape)[i];
    }
    Py_buffer view;
    view.buf = ptr;
    view.obj = nullptr;
    view.len = size * itemsize;
    view.readonly = static_cast<int>(readonly);
    view.itemsize = itemsize;
    view.format = const_cast<char *>(format);
    view.ndim = static_cast<int>(ndim);
    view.shape = shape->data();
    view.strides = strides->data();
    view.suboffsets = nullptr;
    view.internal = nullptr;
    PyObject *obj = PyMemoryView_FromBuffer(&view);
    if (!obj) {
        throw error_already_set();
    }
    return memoryview(object(obj, stolen_t{}));
}
/// @endcond

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
