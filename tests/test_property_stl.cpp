#include <pybind11/smart_holder.h>
#include <pybind11/stl.h>

#include "pybind11_tests.h"

#include <cstddef>
#include <vector>

namespace test_class_sh_property_stl {

struct Field {
    explicit Field(int wrapped_int) : wrapped_int{wrapped_int} {}
    int wrapped_int = 100;
};

struct FieldHolder {
    explicit FieldHolder(const Field &fld) : fld{fld} {}
    Field fld = Field{200};
};

struct VectorFieldHolder {
    std::vector<FieldHolder> vec_fld_hld;
    VectorFieldHolder() { vec_fld_hld.push_back(FieldHolder{Field{300}}); }
    void reset_at(std::size_t index, int wrapped_int) {
        if (index < vec_fld_hld.size()) {
            vec_fld_hld[index].fld.wrapped_int = wrapped_int;
        }
    }
};

} // namespace test_class_sh_property_stl

using namespace test_class_sh_property_stl;

PYBIND11_SMART_HOLDER_TYPE_CASTERS(Field)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(FieldHolder)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(VectorFieldHolder)

TEST_SUBMODULE(class_sh_property_stl, m) {
    py::classh<Field>(m, "Field").def_readwrite("wrapped_int", &Field::wrapped_int);

    py::classh<FieldHolder>(m, "FieldHolder").def_readwrite("fld", &FieldHolder::fld);

    py::classh<VectorFieldHolder>(m, "VectorFieldHolder")
        .def(py::init<>())
        .def("reset_at", &VectorFieldHolder::reset_at)
        .def_readwrite("vec_fld_hld_ref", &VectorFieldHolder::vec_fld_hld)
        .def_readwrite("vec_fld_hld_cpy",
                       &VectorFieldHolder::vec_fld_hld,
                       py::return_value_policy::_clif_automatic);
}
