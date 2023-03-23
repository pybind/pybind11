#include "pybind11/smart_holder.h"
#include "pybind11_tests.h"

#include <cstddef>
#include <vector>

namespace test_class_sh_property_non_owning {

struct CoreField {
    CoreField(int int_value = -99) : int_value{int_value} {}
    int int_value;
};

struct DataField {
    DataField(const CoreField &core_fld = CoreField{}) : core_fld{core_fld} {}
    CoreField core_fld;
};

struct DataFieldsHolder {
private:
    std::vector<DataField> vec;

public:
    DataFieldsHolder(std::size_t vec_size) {
        for (std::size_t i = 0; i < vec_size; i++) {
            vec.push_back(DataField{CoreField{13 + static_cast<int>(i) * 11}});
        }
    }

    DataField *vec_at(std::size_t index) {
        if (index >= vec.size()) {
            return nullptr;
        }
        return &vec[index];
    }
};

} // namespace test_class_sh_property_non_owning

using namespace test_class_sh_property_non_owning;

PYBIND11_SMART_HOLDER_TYPE_CASTERS(CoreField)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(DataField)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(DataFieldsHolder)

TEST_SUBMODULE(class_sh_property_non_owning, m) {
    py::classh<CoreField>(m, "CoreField")
        .def(py::init<>())
        .def_readwrite("int_value", &CoreField::int_value);

    py::classh<DataField>(m, "DataField")
        .def(py::init<>())
        .def_readwrite("core_fld", &DataField::core_fld);

    py::classh<DataFieldsHolder>(m, "DataFieldsHolder")
        .def(py::init<std::size_t>())
        .def("vec_at", &DataFieldsHolder::vec_at, py::return_value_policy::reference_internal);
}
