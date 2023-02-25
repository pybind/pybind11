#include <pybind11/smart_holder.h>
#include <pybind11/stl.h>

#include "pybind11_tests.h"

#include <vector>

struct Value {};

struct ValueHolder {
    Value held_value;
};

struct VectorValueHolder {
    std::vector<ValueHolder> vec_val_hld;
    VectorValueHolder() { vec_val_hld.push_back(ValueHolder{Value{}}); }
};

#define PYBIND11_SH_DBG_CLASSH_ON
#ifdef PYBIND11_SH_DBG_CLASSH_ON
#    define PY_CLASS py::classh
PYBIND11_SMART_HOLDER_TYPE_CASTERS(Value)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(ValueHolder)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(VectorValueHolder)
#else
#    define PY_CLASS py::class_
#endif

TEST_SUBMODULE(non_owning_holder_issue, m) {
    PY_CLASS<Value>(m, "Value");

    PY_CLASS<ValueHolder>(m, "ValueHolder")
        .def(py::init<>())
        .def_readwrite("held_value", &ValueHolder::held_value);

    PY_CLASS<VectorValueHolder>(m, "VectorValueHolder")
        .def(py::init<>())
        .def_readwrite("vec_val_hld", &VectorValueHolder::vec_val_hld);
}
