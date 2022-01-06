#include "pybind11_tests.h"

#include "pybind11/smart_holder.h"

#include <memory>

namespace test_class_sh_property {

struct ClassicField {
    int num = -88;
};

struct ClassicOuter {
    ClassicField *m_mptr = nullptr;
};

struct Field {
    int num = -99;
};

struct Outer {
    Field m_valu;
    Field *m_mptr = nullptr;
    std::unique_ptr<Field> m_uqmp;
    std::shared_ptr<Field> m_shmp;
};

inline void DisownOuter(std::unique_ptr<Outer>) {}

} // namespace test_class_sh_property

PYBIND11_SMART_HOLDER_TYPE_CASTERS(test_class_sh_property::Field)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(test_class_sh_property::Outer)

TEST_SUBMODULE(class_sh_property, m) {
    using namespace test_class_sh_property;

    py::class_<ClassicField, std::unique_ptr<ClassicField>>(m, "ClassicField")
        .def(py::init<>())                        //
        .def_readwrite("num", &ClassicField::num) //
        ;

    py::class_<ClassicOuter, std::unique_ptr<ClassicOuter>>(m, "ClassicOuter")
        .def(py::init<>())                              //
        .def_readwrite("m_mptr", &ClassicOuter::m_mptr) //
        ;

    py::classh<Field>(m, "Field")          //
        .def(py::init<>())                 //
        .def_readwrite("num", &Field::num) //
        ;

    py::classh<Outer>(m, "Outer")                //
        .def(py::init<>())                       //
        .def_readwrite("m_valu", &Outer::m_valu) //
        .def_property(                           //
            "m_mptr",
            [](const std::shared_ptr<Outer> &self) {
                return std::shared_ptr<Field>(self, self->m_mptr);
            },
            [](Outer &self, Field *mptr) { self.m_mptr = mptr; })
        .def_property_readonly( //
            "m_uqmp",
            [](const std::shared_ptr<Outer> &self) {
                return std::shared_ptr<Field>(self, self->m_uqmp.get());
            })
        .def_property( //
            "m_uqmp_disown",
            [](const std::shared_ptr<Outer> &self) {
                return std::unique_ptr<Field>(std::move(self->m_uqmp));
            },
            [](Outer &self, std::unique_ptr<Field> uqmp) {
                self.m_uqmp = std::move(uqmp); //
            })
        .def_readwrite("m_shmp", &Outer::m_shmp) //
        ;

    m.def("DisownOuter", DisownOuter);
}
