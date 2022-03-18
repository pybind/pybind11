#include <pybind11/smart_holder.h>

#include "pybind11_tests.h"

#include <memory>

namespace pybind11_tests {
namespace class_sh_void_ptr_capsule {

struct Valid {};

struct NoConversion {};

struct NoCapsuleReturned {};

struct AsAnotherObject {};

// Create a void pointer capsule for test. The encapsulated object does not
// matter for this test case.
PyObject *create_test_void_ptr_capsule() {
    static int just_to_have_something_to_point_to = 0;
    return PyCapsule_New(static_cast<void *>(&just_to_have_something_to_point_to), "int", nullptr);
}

int get_from_valid_capsule(const Valid *) { return 1; }

int get_from_shared_ptr_valid_capsule(const std::shared_ptr<Valid> &) { return 2; }

int get_from_unique_ptr_valid_capsule(std::unique_ptr<Valid>) { return 3; }

int get_from_no_conversion_capsule(const NoConversion *) { return 4; }

int get_from_no_capsule_returned(const NoCapsuleReturned *) { return 5; }

// https://github.com/pybind/pybind11/issues/3788
struct TypeWithGetattr {
    TypeWithGetattr() = default;
    int get_42() const { return 42; }
};

// https://github.com/pybind/pybind11/issues/3804
struct Base1 {
    int a1{};
};
struct Base2 {
    int a2{};
};

struct Base12 : Base1, Base2 {
    int foo() const { return 0; }
};

struct Derived1 : Base12 {
    int bar() const { return 1; }
};

struct Derived2 : Base12 {
    int bar() const { return 2; }
};

} // namespace class_sh_void_ptr_capsule
} // namespace pybind11_tests

PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_void_ptr_capsule::Valid)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_void_ptr_capsule::TypeWithGetattr)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_void_ptr_capsule::Base1)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_void_ptr_capsule::Base2)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_void_ptr_capsule::Base12)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_void_ptr_capsule::Derived1)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_void_ptr_capsule::Derived2)

TEST_SUBMODULE(class_sh_void_ptr_capsule, m) {
    using namespace pybind11_tests::class_sh_void_ptr_capsule;

    py::classh<Valid>(m, "Valid");

    m.def("get_from_valid_capsule", &get_from_valid_capsule);
    m.def("get_from_shared_ptr_valid_capsule", &get_from_shared_ptr_valid_capsule);
    m.def("get_from_unique_ptr_valid_capsule", &get_from_unique_ptr_valid_capsule);
    m.def("get_from_no_conversion_capsule", &get_from_no_conversion_capsule);
    m.def("get_from_no_capsule_returned", &get_from_no_capsule_returned);
    m.def("create_test_void_ptr_capsule", []() {
        PyObject *capsule = create_test_void_ptr_capsule();
        return pybind11::reinterpret_steal<py::capsule>(capsule);
    });

    py::classh<TypeWithGetattr>(m, "TypeWithGetattr")
        .def(py::init<>())
        .def("get_42", &TypeWithGetattr::get_42)
        .def("__getattr__",
             [](TypeWithGetattr &, const std::string &key) { return "GetAttr: " + key; });

    py::classh<Base1>(m, "Base1");
    py::classh<Base2>(m, "Base2");

    py::classh<Base12, Base1, Base2>(m, "Base12")
        .def(py::init<>())
        .def("foo", &Base12::foo)
        .def("__getattr__",
             [](Base12 &, const std::string &key) { return "Base GetAttr: " + key; });

    py::classh<Derived1, Base12>(m, "Derived1").def(py::init<>()).def("bar", &Derived1::bar);

    py::classh<Derived2, Base12>(m, "Derived2").def(py::init<>()).def("bar", &Derived2::bar);
}
