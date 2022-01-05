#include "pybind11_tests.h"

#include "pybind11/smart_holder.h"

#include <memory>

namespace test_class_sh_property {

struct Field {
    int num = -99;
};

struct Outer {
    Field m_valu;
    Field *m_mptr;
    std::unique_ptr<Field> m_uqmp;
    std::shared_ptr<Field> m_shmp;
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

    py::classh<Outer>(m, "Outer")                //
        .def(py::init<>())                       //
        .def_readwrite("m_valu", &Outer::m_valu) //
        // .def_readwrite("m_mptr", &Outer::m_mptr) //
        // .def_readwrite("m_uqmp", &Outer::m_uqmp) //
        .def_readwrite("m_shmp", &Outer::m_shmp) //
        ;

    m.def("DisownOuter", DisownOuter);
}
