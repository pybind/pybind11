#include "init.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

PYBIND11_NAMESPACE_BEGIN(detail)

PYBIND11_NAMESPACE_BEGIN(initimpl)

PYBIND11_INLINE void no_nullptr(void *ptr) {
    if (!ptr) throw type_error("pybind11::init(): factory function returned nullptr");
}

PYBIND11_NAMESPACE_END(initimpl)

PYBIND11_NAMESPACE_END(detail)

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
