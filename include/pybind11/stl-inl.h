#include "stl.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

PYBIND11_INLINE std::ostream &operator<<(std::ostream &os, const handle &obj) {
    os << (std::string) str(obj);
    return os;
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
