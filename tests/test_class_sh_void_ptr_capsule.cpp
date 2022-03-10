#include <pybind11/smart_holder.h>

#include "pybind11_tests.h"

#include <memory>

namespace pybind11_tests {
namespace class_sh_void_ptr_capsule {

// Conveniently, the helper serves to keep track of `capsule_generated`.
struct HelperBase {
    HelperBase() = default;
    HelperBase(const HelperBase &) = delete;
    virtual ~HelperBase() = default;

    bool capsule_generated = false;
    virtual int get() const { return 100; }
};

struct Valid : public HelperBase {
    int get() const override { return 101; }

    PyObject *as_pybind11_tests_class_sh_void_ptr_capsule_Valid() {
        void *vptr = dynamic_cast<void *>(this);
        capsule_generated = true;
        // We assume vptr out lives the capsule, so we use nullptr for the
        // destructor.
        return PyCapsule_New(vptr, "::pybind11_tests::class_sh_void_ptr_capsule::Valid", nullptr);
    }
};

struct NoConversion : public HelperBase {
    int get() const override { return 102; }
};

struct NoCapsuleReturned : public HelperBase {
    int get() const override { return 103; }

    PyObject *as_pybind11_tests_class_sh_void_ptr_capsule_NoCapsuleReturned() {
        capsule_generated = true;
        Py_XINCREF(Py_None);
        return Py_None;
    }
};

struct AsAnotherObject : public HelperBase {
    int get() const override { return 104; }

    PyObject *as_pybind11_tests_class_sh_void_ptr_capsule_Valid() {
        void *vptr = dynamic_cast<void *>(this);
        capsule_generated = true;
        // We assume vptr out lives the capsule, so we use nullptr for the
        // destructor.
        return PyCapsule_New(vptr, "::pybind11_tests::class_sh_void_ptr_capsule::Valid", nullptr);
    }
};

// https://github.com/pybind/pybind11/issues/3788
struct TypeWithGetattr {
    TypeWithGetattr() = default;
    int get_42() const { return 42; }
};

int get_from_valid_capsule(const Valid *c) { return c->get(); }

int get_from_shared_ptr_valid_capsule(const std::shared_ptr<Valid> &c) { return c->get(); }

int get_from_unique_ptr_valid_capsule(std::unique_ptr<Valid> c) { return c->get(); }

int get_from_no_conversion_capsule(const NoConversion *c) { return c->get(); }

int get_from_no_capsule_returned(const NoCapsuleReturned *c) { return c->get(); }

} // namespace class_sh_void_ptr_capsule
} // namespace pybind11_tests

PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_void_ptr_capsule::HelperBase)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_void_ptr_capsule::Valid)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_void_ptr_capsule::NoConversion)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_void_ptr_capsule::NoCapsuleReturned)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_void_ptr_capsule::AsAnotherObject)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_void_ptr_capsule::TypeWithGetattr)

TEST_SUBMODULE(class_sh_void_ptr_capsule, m) {
    using namespace pybind11_tests::class_sh_void_ptr_capsule;

    py::classh<HelperBase>(m, "HelperBase")
        .def(py::init<>())
        .def("get", &HelperBase::get)
        .def_readonly("capsule_generated", &HelperBase::capsule_generated);

    py::classh<Valid, HelperBase>(m, "Valid")
        .def(py::init<>())
        .def("as_pybind11_tests_class_sh_void_ptr_capsule_Valid", [](Valid &self) {
            PyObject *capsule = self.as_pybind11_tests_class_sh_void_ptr_capsule_Valid();
            return pybind11::reinterpret_steal<py::capsule>(capsule);
        });

    py::classh<NoConversion, HelperBase>(m, "NoConversion").def(py::init<>());

    py::classh<NoCapsuleReturned, HelperBase>(m, "NoCapsuleReturned")
        .def(py::init<>())
        .def("as_pybind11_tests_class_sh_void_ptr_capsule_NoCapsuleReturned",
             [](NoCapsuleReturned& self) {
                 PyObject *capsule
                     = self.as_pybind11_tests_class_sh_void_ptr_capsule_NoCapsuleReturned();
                 return pybind11::reinterpret_steal<py::capsule>(capsule);
             });

    py::classh<AsAnotherObject, HelperBase>(m, "AsAnotherObject")
        .def(py::init<>())
        .def("as_pybind11_tests_class_sh_void_ptr_capsule_Valid",
             [](AsAnotherObject& self) {
            PyObject *capsule = self.as_pybind11_tests_class_sh_void_ptr_capsule_Valid();
            return pybind11::reinterpret_steal<py::capsule>(capsule);
        });

    m.def("get_from_valid_capsule", &get_from_valid_capsule);
    m.def("get_from_shared_ptr_valid_capsule", &get_from_shared_ptr_valid_capsule);
    m.def("get_from_unique_ptr_valid_capsule", &get_from_unique_ptr_valid_capsule);
    m.def("get_from_no_conversion_capsule", &get_from_no_conversion_capsule);
    m.def("get_from_no_capsule_returned", &get_from_no_capsule_returned);

    py::classh<TypeWithGetattr>(m, "TypeWithGetattr")
        .def(py::init<>())
        .def("get_42", &TypeWithGetattr::get_42)
        .def("__getattr__",
             [](TypeWithGetattr &, const std::string &key) { return "GetAttr: " + key; });
}
