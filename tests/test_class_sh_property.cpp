#include "pybind11_tests.h"

#include "pybind11/smart_holder.h"

#include <memory>

namespace test_class_sh_property {

struct Field {
    int num = -99;
};

struct Outer {
    Field m_val;
    std::shared_ptr<Field> m_sh_ptr;
};

inline void DisownOuter(std::unique_ptr<Outer>) {}

} // namespace test_class_sh_property

PYBIND11_SMART_HOLDER_TYPE_CASTERS(test_class_sh_property::Field)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(test_class_sh_property::Outer)

TEST_SUBMODULE(class_sh_property, m) {
    using namespace test_class_sh_property;

    py::classh<Field>(m, "Field")          //
        .def(py::init<>())                 //
        .def_readwrite("num", &Field::num) //
        ;

    py::classh<Outer>(m, "Outer")                    //
        .def(py::init<>())                           //
        .def_readwrite("m_val", &Outer::m_val)       //
        .def_readwrite("m_sh_ptr", &Outer::m_sh_ptr) //
        ;

    m.def("DisownOuter", DisownOuter);
}
