#include "pybind11_tests.h"

#include <cstddef>
#include <vector>

namespace test_class_sh_property_non_owning {

struct CoreField {
    explicit CoreField(int int_value = -99) : int_value{int_value} {}
    int int_value;
};

struct DataField {
    DataField(int i_value, int i_shared, int i_unique)
        : core_fld_value{i_value}, core_fld_shared_ptr{new CoreField{i_shared}},
          core_fld_raw_ptr{core_fld_shared_ptr.get()},
          core_fld_unique_ptr{new CoreField{i_unique}} {}
    CoreField core_fld_value;
    std::shared_ptr<CoreField> core_fld_shared_ptr;
    CoreField *core_fld_raw_ptr;
    std::unique_ptr<CoreField> core_fld_unique_ptr;
};

struct DataFieldsHolder {
private:
    std::vector<DataField> vec;

public:
    explicit DataFieldsHolder(std::size_t vec_size) {
        for (std::size_t i = 0; i < vec_size; i++) {
            int i11 = static_cast<int>(i) * 11;
            vec.emplace_back(13 + i11, 14 + i11, 15 + i11);
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

TEST_SUBMODULE(class_sh_property_non_owning, m) {
    py::classh<CoreField>(m, "CoreField").def_readwrite("int_value", &CoreField::int_value);

    py::classh<DataField>(m, "DataField")
        .def_readonly("core_fld_value_ro", &DataField::core_fld_value)
        .def_readwrite("core_fld_value_rw", &DataField::core_fld_value)
        .def_readonly("core_fld_shared_ptr_ro", &DataField::core_fld_shared_ptr)
        .def_readwrite("core_fld_shared_ptr_rw", &DataField::core_fld_shared_ptr)
        .def_readonly("core_fld_raw_ptr_ro", &DataField::core_fld_raw_ptr)
        .def_readwrite("core_fld_raw_ptr_rw", &DataField::core_fld_raw_ptr)
        .def_readwrite("core_fld_unique_ptr_rw", &DataField::core_fld_unique_ptr);

    py::classh<DataFieldsHolder>(m, "DataFieldsHolder")
        .def(py::init<std::size_t>())
        .def("vec_at", &DataFieldsHolder::vec_at, py::return_value_policy::reference_internal);
}
