/*
    pybind11/buffer_info.h: Python buffer object interface

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"

NAMESPACE_BEGIN(pybind11)

/// Information record describing a Python buffer object
struct buffer_info {
    void *ptr = nullptr;         // Pointer to the underlying storage
    size_t itemsize = 0;         // Size of individual items in bytes
    size_t size = 0;             // Total number of entries
    std::string format;          // For homogeneous buffers, this should be set to format_descriptor<T>::format()
    size_t ndim = 0;             // Number of dimensions
    std::vector<size_t> shape;   // Shape of the tensor (1 entry per dimension)
    std::vector<size_t> strides; // Number of entries between adjacent entries (for each per dimension)

    buffer_info() { }

    buffer_info(void *ptr, size_t itemsize, const std::string &format, size_t ndim,
                detail::any_container<size_t> shape_in, detail::any_container<size_t> strides_in)
    : ptr(ptr), itemsize(itemsize), size(1), format(format), ndim(ndim),
      shape(std::move(shape_in)), strides(std::move(strides_in)) {
        if (ndim != shape.size() || ndim != strides.size())
            pybind11_fail("buffer_info: ndim doesn't match shape and/or strides length");
        for (size_t i = 0; i < ndim; ++i)
            size *= shape[i];
    }

    buffer_info(void *ptr, size_t itemsize, const std::string &format, size_t size)
    : buffer_info(ptr, itemsize, format, 1, {size}, {itemsize}) { }

    explicit buffer_info(Py_buffer *view, bool ownview = true)
    : buffer_info(view->buf, (size_t) view->itemsize, view->format, (size_t) view->ndim,
            {view->shape, view->shape + view->ndim}, {view->strides, view->strides + view->ndim}) {
        this->view = view;
        this->ownview = ownview;
    }

    buffer_info(const buffer_info &) = delete;
    buffer_info& operator=(const buffer_info &) = delete;

    buffer_info(buffer_info &&other) {
        (*this) = std::move(other);
    }

    buffer_info& operator=(buffer_info &&rhs) {
        ptr = rhs.ptr;
        itemsize = rhs.itemsize;
        size = rhs.size;
        format = std::move(rhs.format);
        ndim = rhs.ndim;
        shape = std::move(rhs.shape);
        strides = std::move(rhs.strides);
        std::swap(view, rhs.view);
        std::swap(ownview, rhs.ownview);
        return *this;
    }

    ~buffer_info() {
        if (view && ownview) { PyBuffer_Release(view); delete view; }
    }

private:
    Py_buffer *view = nullptr;
    bool ownview = false;
};

NAMESPACE_BEGIN(detail)

template <typename T, typename SFINAE = void> struct compare_buffer_info {
    static bool compare(const buffer_info& b) {
        return b.format == format_descriptor<T>::format() && b.itemsize == sizeof(T);
    }
};

template <typename T> struct compare_buffer_info<T, detail::enable_if_t<std::is_integral<T>::value>> {
    static bool compare(const buffer_info& b) {
        return b.itemsize == sizeof(T) && (b.format == format_descriptor<T>::value ||
            ((sizeof(T) == sizeof(long)) && b.format == (std::is_unsigned<T>::value ? "L" : "l")) ||
            ((sizeof(T) == sizeof(size_t)) && b.format == (std::is_unsigned<T>::value ? "N" : "n")));
    }
};

NAMESPACE_END(detail)
NAMESPACE_END(pybind11)
