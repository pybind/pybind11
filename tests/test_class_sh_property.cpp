#include "pybind11_tests.h"

#include "pybind11/smart_holder.h"

#include <memory>

namespace test_class_sh_property {

struct Inner {
    int value = -99;
};

struct Outer {
    Inner field;
};

inline void DisownOuter(std::unique_ptr<Outer>) {}

} // namespace test_class_sh_property

PYBIND11_SMART_HOLDER_TYPE_CASTERS(test_class_sh_property::Inner)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(test_class_sh_property::Outer)

TEST_SUBMODULE(class_sh_property, m) {
    using namespace test_class_sh_property;

    py::classh<Inner>(m, "Inner")              //
        .def(py::init<>())                     //
        .def_readwrite("value", &Inner::value) //
        ;

    py::classh<Outer>(m, "Outer") //
        .def(py::init<>())        //
        .def_property(
            "field", //
            [](const std::shared_ptr<Outer> &self) {
                // Emulating PyCLIF approach:
                // https://github.com/google/clif/blob/c371a6d4b28d25d53a16e6d2a6d97305fb1be25a/clif/python/instance.h#L233
                return std::shared_ptr<Inner>(self, &self->field);
            },                                                          //
            [](Outer &self, const Inner &field) { self.field = field; } //
            )                                                           //
        ;

    m.def("DisownOuter", DisownOuter);
}
