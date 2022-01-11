#include "pybind11_tests.h"

#include "pybind11/smart_holder.h"

#include <memory>

namespace test_class_sh_property {

struct ClassicField {
    int num = -88;
};

struct ClassicOuter {
    ClassicField *m_mptr = nullptr;
    const ClassicField *m_cptr = nullptr;
};

struct Field {
    int num = -99;
};

struct Outer {
    Field m_valu;
    Field *m_mptr = nullptr;
    const Field *m_cptr = nullptr;
    std::unique_ptr<Field> m_uqmp;
    std::unique_ptr<const Field> m_uqcp;
    std::shared_ptr<Field> m_shmp;
    std::shared_ptr<const Field> m_shcp;
};

inline void DisownOuter(std::unique_ptr<Outer>) {}

} // namespace test_class_sh_property

PYBIND11_TYPE_CASTER_BASE_HOLDER(test_class_sh_property::ClassicField,
                                 std::unique_ptr<test_class_sh_property::ClassicField>)
PYBIND11_TYPE_CASTER_BASE_HOLDER(test_class_sh_property::ClassicOuter,
                                 std::unique_ptr<test_class_sh_property::ClassicOuter>)

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
        .def_readwrite("m_cptr", &ClassicOuter::m_cptr) //
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
        .def_property( //
            "m_cptr",
            [](const std::shared_ptr<Outer> &self) {
                return std::shared_ptr<const Field>(self, self->m_cptr);
            },
            [](Outer &self, const Field *cptr) { self.m_cptr = cptr; })
        .def_property( //
            "m_uqmp_disown",
            [](const std::shared_ptr<Outer> &self) {
                return std::unique_ptr<Field>(std::move(self->m_uqmp));
            },
            [](Outer &self, std::unique_ptr<Field> uqmp) {
                self.m_uqmp = std::move(uqmp); //
            })
        .def_property( //
            "m_uqcp_disown",
            [](const std::shared_ptr<Outer> &self) {
                return std::unique_ptr<const Field>(std::move(self->m_uqcp));
            },
            [](Outer &self, std::unique_ptr<const Field> uqcp) {
                self.m_uqcp = std::move(uqcp); //
            })
        .def_readwrite("m_shmp", &Outer::m_shmp) //
        .def_readwrite("m_shcp", &Outer::m_shcp) //
        ;

    m.def("DisownOuter", DisownOuter);
}
