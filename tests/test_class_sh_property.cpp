// The compact 4-character naming matches that in test_class_sh_basic.cpp
// Variable names are intentionally terse, to not distract from the more important C++ type names:
// valu(e), ref(erence), ptr or p (pointer), r = rvalue, m = mutable, c = const,
// sh = shared_ptr, uq = unique_ptr.

#include "pybind11/smart_holder.h"
#include "pybind11_tests.h"

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
        .def(py::init<>())
        .def_readwrite("num", &ClassicField::num);

    py::class_<ClassicOuter, std::unique_ptr<ClassicOuter>>(m, "ClassicOuter")
        .def(py::init<>())
        .def_readonly("m_mptr_readonly", &ClassicOuter::m_mptr)
        .def_readwrite("m_mptr_readwrite", &ClassicOuter::m_mptr)
        .def_readwrite("m_cptr_readonly", &ClassicOuter::m_cptr)
        .def_readwrite("m_cptr_readwrite", &ClassicOuter::m_cptr);

    py::classh<Field>(m, "Field").def(py::init<>()).def_readwrite("num", &Field::num);

    py::classh<Outer>(m, "Outer")
        .def(py::init<>())

        .def_readonly("m_valu_readonly", &Outer::m_valu)
        .def_readwrite("m_valu_readwrite", &Outer::m_valu)

        .def_readonly("m_mptr_readonly", &Outer::m_mptr)
        .def_readwrite("m_mptr_readwrite", &Outer::m_mptr)
        .def_readonly("m_cptr_readonly", &Outer::m_cptr)
        .def_readwrite("m_cptr_readwrite", &Outer::m_cptr)

        // .def_readonly("m_uqmp_readonly", &Outer::m_uqmp) // Custom compilation Error.
        .def_readwrite("m_uqmp_readwrite", &Outer::m_uqmp)
        // .def_readonly("m_uqcp_readonly", &Outer::m_uqcp) // Custom compilation Error.
        .def_readwrite("m_uqcp_readwrite", &Outer::m_uqcp)

        .def_readwrite("m_shmp_readonly", &Outer::m_shmp)
        .def_readwrite("m_shmp_readwrite", &Outer::m_shmp)
        .def_readwrite("m_shcp_readonly", &Outer::m_shcp)
        .def_readwrite("m_shcp_readwrite", &Outer::m_shcp);

    m.def("DisownOuter", DisownOuter);
}
