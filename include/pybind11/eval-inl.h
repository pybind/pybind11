#include "eval.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

PYBIND11_INLINE void exec(str expr, object global, object local) {
    eval<eval_statements>(expr, global, local);
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
