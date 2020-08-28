#include "buffer_info.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

PYBIND11_INLINE buffer_info::buffer_info(void *ptr, ssize_t itemsize, const std::string &format, ssize_t ndim,
            detail::any_container<ssize_t> shape_in, detail::any_container<ssize_t> strides_in, bool readonly)
: ptr(ptr), itemsize(itemsize), size(1), format(format), ndim(ndim),
    shape(std::move(shape_in)), strides(std::move(strides_in)), readonly(readonly) {
    if (ndim != (ssize_t) shape.size() || ndim != (ssize_t) strides.size())
        pybind11_fail("buffer_info: ndim doesn't match shape and/or strides length");
    for (size_t i = 0; i < (size_t) ndim; ++i)
        size *= shape[i];
}

PYBIND11_INLINE buffer_info::buffer_info(Py_buffer *view, bool ownview)
: buffer_info(view->buf, view->itemsize, view->format, view->ndim,
        {view->shape, view->shape + view->ndim}, {view->strides, view->strides + view->ndim}, view->readonly) {
    this->m_view = view;
    this->ownview = ownview;
}

PYBIND11_INLINE buffer_info::buffer_info(buffer_info &&other) {
    (*this) = std::move(other);
}

PYBIND11_INLINE buffer_info& buffer_info::operator=(buffer_info &&rhs) {
    ptr = rhs.ptr;
    itemsize = rhs.itemsize;
    size = rhs.size;
    format = std::move(rhs.format);
    ndim = rhs.ndim;
    shape = std::move(rhs.shape);
    strides = std::move(rhs.strides);
    std::swap(m_view, rhs.m_view);
    std::swap(ownview, rhs.ownview);
    readonly = rhs.readonly;
    return *this;
}

PYBIND11_INLINE buffer_info::~buffer_info() {
    if (m_view && ownview) { PyBuffer_Release(m_view); delete m_view; }
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
