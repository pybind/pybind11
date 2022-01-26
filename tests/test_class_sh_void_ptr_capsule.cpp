#include "pybind11_tests.h"

#include <pybind11/smart_holder.h>

#include <memory>

namespace pybind11_tests {
namespace class_sh_void_ptr_capsule {

// Without the helper we will run into a type_caster::load recursion.
// This is because whenever the type_caster::load is called, it checks
// whether the object defines an `as_` method that returns the void pointer
// capsule. If yes, it calls the method. But in the following testcases, those
// `as_` methods are defined with pybind11, which implicitly takes the object
// itself as the first parameter. Therefore calling those methods causes loading
// the object again, which causes infinite recursion.
// This test is unusual in the sense that the void pointer capsules are meant to
// be provided by objects wrapped with systems other than pybind11
// (i.e. having to avoid the recursion is an artificial problem, not the norm).
// Conveniently, the helper also serves to keep track of `capsule_generated`.
struct HelperBase {
    HelperBase() = default;
    HelperBase(const HelperBase &) = delete;
    virtual ~HelperBase() = default;

    bool capsule_generated = false;
    virtual int get() const { return 100; }
};

struct Valid: public HelperBase {
    int get() const override { return 101; }

    PyObject* as_pybind11_tests_class_sh_void_ptr_capsule_Valid() {
      void* vptr = dynamic_cast<void*>(this);
      capsule_generated = true;
      // We assume vptr out lives the capsule, so we use nullptr for the
      // destructor.
      return PyCapsule_New(
          vptr, "::pybind11_tests::class_sh_void_ptr_capsule::Valid",
          nullptr);
    }
};

struct NoConversion: public HelperBase {
    int get() const override { return 102; }
};

struct NoCapsuleReturned: public HelperBase {
    int get() const override { return 103; }

    PyObject* as_pybind11_tests_class_sh_void_ptr_capsule_NoCapsuleReturned() {
      capsule_generated = true;
      Py_XINCREF(Py_None);
      return Py_None;
    }
};

struct AsAnotherObject: public HelperBase {
    int get() const override { return 104; }

    PyObject* as_pybind11_tests_class_sh_void_ptr_capsule_Valid() {
      void* vptr = dynamic_cast<void*>(this);
      capsule_generated = true;
      // We assume vptr out lives the capsule, so we use nullptr for the
      // destructor.
      return PyCapsule_New(
          vptr, "::pybind11_tests::class_sh_void_ptr_capsule::Valid",
          nullptr);
    }
};

int get_from_valid_capsule(const Valid* c) {
  return c->get();
}

int get_from_shared_ptr_valid_capsule(const std::shared_ptr<Valid> &c) {
  return c->get();
}

int get_from_unique_ptr_valid_capsule(std::unique_ptr<Valid> c) {
  return c->get();
}

int get_from_no_conversion_capsule(const NoConversion* c) {
  return c->get();
}

int get_from_no_capsule_returned(const NoCapsuleReturned* c) {
  return c->get();
}

} // namespace class_sh_void_ptr_capsule
} // namespace pybind11_tests

PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_void_ptr_capsule::HelperBase)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_void_ptr_capsule::Valid)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_void_ptr_capsule::NoConversion)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_void_ptr_capsule::NoCapsuleReturned)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_void_ptr_capsule::AsAnotherObject)

TEST_SUBMODULE(class_sh_void_ptr_capsule, m) {
    using namespace pybind11_tests::class_sh_void_ptr_capsule;

    py::classh<HelperBase>(m, "HelperBase")
        .def(py::init<>())
        .def("get", &HelperBase::get)
        .def_readonly("capsule_generated", &HelperBase::capsule_generated);

    py::classh<Valid, HelperBase>(m, "Valid")
        .def(py::init<>())
        .def("as_pybind11_tests_class_sh_void_ptr_capsule_Valid",
             [](HelperBase* self) {
          auto obj = dynamic_cast<Valid *>(self);
          assert(obj != nullptr);
          PyObject* capsule = obj->as_pybind11_tests_class_sh_void_ptr_capsule_Valid();
          return pybind11::reinterpret_steal<py::capsule>(capsule);
        });

    py::classh<NoConversion, HelperBase>(m, "NoConversion")
        .def(py::init<>());

    py::classh<NoCapsuleReturned, HelperBase>(m, "NoCapsuleReturned")
        .def(py::init<>())
        .def("as_pybind11_tests_class_sh_void_ptr_capsule_NoCapsuleReturned",
             [](HelperBase* self) {
          auto obj = dynamic_cast<NoCapsuleReturned *>(self);
          assert(obj != nullptr);
          PyObject* capsule = obj->as_pybind11_tests_class_sh_void_ptr_capsule_NoCapsuleReturned();
          return pybind11::reinterpret_steal<py::capsule>(capsule);
        });

    py::classh<AsAnotherObject, HelperBase>(m, "AsAnotherObject")
        .def(py::init<>())
        .def("as_pybind11_tests_class_sh_void_ptr_capsule_Valid",
             [](HelperBase* self) {
          auto obj = dynamic_cast<AsAnotherObject *>(self);
          assert(obj != nullptr);
          PyObject* capsule = obj->as_pybind11_tests_class_sh_void_ptr_capsule_Valid();
          return pybind11::reinterpret_steal<py::capsule>(capsule);
        });

    m.def("get_from_valid_capsule", &get_from_valid_capsule);
    m.def("get_from_shared_ptr_valid_capsule", &get_from_shared_ptr_valid_capsule);
    m.def("get_from_unique_ptr_valid_capsule", &get_from_unique_ptr_valid_capsule);
    m.def("get_from_no_conversion_capsule", &get_from_no_conversion_capsule);
    m.def("get_from_no_capsule_returned", &get_from_no_capsule_returned);
}
