#include "pybind11_tests.h"

#include "pybind11/smart_holder.h"

#include <memory>

namespace {

struct Inner {
    int value = -99;
};

struct Outer {
    Inner field;
};

inline void DisownOuter(std::unique_ptr<Outer>) {}

} // namespace

PYBIND11_SMART_HOLDER_TYPE_CASTERS(Inner)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(Outer)

TEST_SUBMODULE(class_sh_property, m) {
    py::classh<Inner>(m, "Inner")              //
        .def_readwrite("value", &Inner::value) //
        ;

    py::classh<Outer>(m, "Outer")              //
        .def(py::init<>())                     //
        .def_readwrite("field", &Outer::field) //
        ;

    m.def("DisownOuter", DisownOuter);
}
