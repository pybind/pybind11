#include "pybind11_tests.h"

#include <cstring>

namespace test_class_sh_property_bakein {

struct WithCharArrayMember {
    WithCharArrayMember() { std::memcpy(char6_member, "Char6", 6); }
    char char6_member[6];
};

struct WithConstCharPtrMember {
    const char *const_char_ptr_member = "ConstChar*";
};

} // namespace test_class_sh_property_bakein

TEST_SUBMODULE(class_sh_property_bakein, m) {
    using namespace test_class_sh_property_bakein;

    py::class_<WithCharArrayMember>(m, "WithCharArrayMember")
        .def(py::init<>())
        .def_readonly("char6_member", &WithCharArrayMember::char6_member);

    py::class_<WithConstCharPtrMember>(m, "WithConstCharPtrMember")
        .def(py::init<>())
        .def_readonly("const_char_ptr_member", &WithConstCharPtrMember::const_char_ptr_member);
}
