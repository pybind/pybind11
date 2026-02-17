#include "pybind11_tests.h"

#include <iostream>

class CrossDSOClass {
public:
    CrossDSOClass() = default;
    virtual ~CrossDSOClass();
    CrossDSOClass(const CrossDSOClass &) = default;
};

CrossDSOClass::~CrossDSOClass() = default;

struct UnrelatedClass {};

TEST_SUBMODULE(class_cross_module_use_after_one_module_dealloc, m) {
    m.def("register_and_instantiate_cross_dso_class", [](const py::module_ &m) {
        py::class_<CrossDSOClass>(m, "CrossDSOClass").def(py::init<>());
        return CrossDSOClass();
    });
    m.def("register_unrelated_class",
          [](const py::module_ &m) { py::class_<UnrelatedClass>(m, "UnrelatedClass"); });
}
