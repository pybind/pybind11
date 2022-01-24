#include "pybind11_tests.h"

#include <pybind11/smart_holder.h>

#include <memory>

namespace pybind11_tests {
namespace class_sh_void_ptr_capsule {

struct CapsuleBase {
    CapsuleBase() = default;
    virtual ~CapsuleBase() = default;

    bool capsule_generated = false;
    virtual int get() const { return 100; }
};

struct Valid: public CapsuleBase {
    int get() const override { return 101; }

    PyObject* as_pybind11_tests_class_sh_void_ptr_capsule_Valid() {
      void* vptr = static_cast<void*>(this);
      capsule_generated = true;
      // We assume vptr out lives the capsule, so we use nullptr for the
      // destructor.
      return PyCapsule_New(
          vptr, "::pybind11_tests::class_sh_void_ptr_capsule::Valid",
          nullptr);
    }
};

struct NoConversion: public CapsuleBase {
    int get() const override { return 102; }
};

struct NoCapsuleReturned: public CapsuleBase {
    int get() const { return 103; }

    PyObject* as_pybind11_tests_class_sh_void_ptr_capsule_NoCapsuleReturned() {
      capsule_generated = true;
      Py_XINCREF(Py_None);
      return Py_None;
    }
};

struct AsAnotherObject: public CapsuleBase {
    int get() const override { return 104; }

    PyObject* as_pybind11_tests_class_sh_void_ptr_capsule_Valid() {
      void* vptr = static_cast<void*>(this);
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

int get_from_shared_ptr_valid_capsule(std::shared_ptr<Valid> c) {
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

PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_void_ptr_capsule::CapsuleBase)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_void_ptr_capsule::Valid)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_void_ptr_capsule::NoConversion)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_void_ptr_capsule::NoCapsuleReturned)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_void_ptr_capsule::AsAnotherObject)

TEST_SUBMODULE(class_sh_void_ptr_capsule, m) {
    using namespace pybind11_tests::class_sh_void_ptr_capsule;

    py::classh<CapsuleBase>(m, "CapsuleBase")
        .def(py::init<>())
        .def("get", &CapsuleBase::get)
        .def_readonly("capsule_generated", &CapsuleBase::capsule_generated);

    py::classh<Valid, CapsuleBase>(m, "Valid")
        .def(py::init<>())
        .def("as_pybind11_tests_class_sh_void_ptr_capsule_Valid",
             [](CapsuleBase* self) {
          Valid *obj = dynamic_cast<Valid *>(self);
          assert(obj != nullptr);
          PyObject* capsule = obj->as_pybind11_tests_class_sh_void_ptr_capsule_Valid();
          return pybind11::reinterpret_steal<py::capsule>(capsule);
        });

    py::classh<NoConversion, CapsuleBase>(m, "NoConversion")
        .def(py::init<>());

    py::classh<NoCapsuleReturned, CapsuleBase>(m, "NoCapsuleReturned")
        .def(py::init<>())
        .def("as_pybind11_tests_class_sh_void_ptr_capsule_NoCapsuleReturned",
             [](CapsuleBase* self) {
          NoCapsuleReturned *obj = dynamic_cast<NoCapsuleReturned *>(self);
          assert(obj != nullptr);
          PyObject* capsule = obj->as_pybind11_tests_class_sh_void_ptr_capsule_NoCapsuleReturned();
          return pybind11::reinterpret_steal<py::capsule>(capsule);
        });

    py::classh<AsAnotherObject, CapsuleBase>(m, "AsAnotherObject")
        .def(py::init<>())
        .def("as_pybind11_tests_class_sh_void_ptr_capsule_Valid",
             [](CapsuleBase* self) {
          AsAnotherObject *obj = dynamic_cast<AsAnotherObject *>(self);
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
