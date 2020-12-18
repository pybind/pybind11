#include "pybind11_tests.h"

#include <iostream>
#include <memory>

namespace pybind11_tests {
namespace smart_ptr_base_derived {

inline void to_cout(std::string text) { std::cout << text << std::endl; }

class cbase {
 public:
  int get_int() const { return 90146438; }
};

class cderived : public cbase {
 public:
  // Printing from constructor & destructor for simple external validation.
  cderived() {
    std::cout << std::endl << "cderived+" << std::endl;
  }
  ~cderived() {
    std::cout << std::endl << "cderived-" << std::endl;
  }
  int get_int() const { return 31607978; }
  int base_get_int(const cbase& base) { return get_int() + base.get_int(); }
};

class vbase {
 public:
  virtual ~vbase() {}
  virtual int get_int() const = 0;
};

class vderived : public vbase {
 public:
  // Printing from constructor & destructor for simple external validation.
  vderived() {
    std::cout << std::endl << "vderived+" << std::endl;
  }
  ~vderived() {
    std::cout << std::endl << "vderived-" << std::endl;
  }
  int get_int() const override { return 29852452; }
  int base_get_int(const vbase& base) { return get_int() + base.get_int(); }
};

class vrederived : public vderived {};

inline std::unique_ptr<cbase>
make_unique_cderived_up_cast() {
  // Undefined Behavior (pure C++ problem, NOT a pybind11 problem):
  // cderived destructor does not run.
  return std::unique_ptr<cderived>(new cderived);
}

inline std::shared_ptr<cderived>
make_shared_cderived(bool use_custom_deleter = false) {
  if (use_custom_deleter) {
    return std::shared_ptr<cderived>(
        new cderived, [](cderived *p) { delete p; });
  }
  return std::shared_ptr<cderived>(new cderived);
}

inline std::shared_ptr<cbase>
make_shared_cderived_up_cast(bool use_custom_deleter = false) {
  return make_shared_cderived(use_custom_deleter);
}

inline int pass_unique_cbase(std::unique_ptr<cbase> cb) {
  return cb->get_int();
}

inline int pass_shared_cbase(std::shared_ptr<cbase> cb) {
  return cb->get_int();
}

inline int pass_shared_cderived(std::shared_ptr<cderived> cd) {
  return cd->get_int();
}

inline std::unique_ptr<vbase>
make_unique_vderived_up_cast() {
  // Well-defined behavior because vderived has a virtual destructor.
  return std::unique_ptr<vderived>(new vderived);
}

inline std::shared_ptr<vderived>
make_shared_vderived(bool use_custom_deleter = false) {
  if (use_custom_deleter) {
    return std::shared_ptr<vderived>(
        new vderived, [](vderived *p) { delete p; });
  }
  return std::shared_ptr<vderived>(new vderived);
}

inline std::shared_ptr<vbase>
make_shared_vderived_up_cast(bool use_custom_deleter = false) {
  return make_shared_vderived(use_custom_deleter);
}

inline int pass_unique_vbase(std::unique_ptr<vbase> vb) {
  return vb->get_int();
}

inline int pass_shared_vbase(std::shared_ptr<vbase> vb) {
  return vb->get_int();
}

inline int pass_shared_vderived(std::shared_ptr<vderived> vd) {
  return vd->get_int();
}

inline int pass_shared_vrederived(std::shared_ptr<vrederived> vr) {
  return vr->get_int();
}

TEST_SUBMODULE(smart_ptr_base_derived, m) {
    m.def("to_cout", to_cout);

    py::class_<cbase, std::shared_ptr<cbase>>(m, "cbase")
        .def(py::init<>())
        .def("get_int", &cbase::get_int);

    py::class_<cderived, cbase, std::shared_ptr<cderived>>(m, "cderived")
        .def(py::init<>())
        .def("get_int", &cderived::get_int);

    py::class_<vbase, std::shared_ptr<vbase>>(m, "vbase")
        .def("get_int", &vbase::get_int);

    py::class_<vderived, vbase, std::shared_ptr<vderived>>(m, "vderived")
        .def(py::init<>());

    py::class_<vrederived, vderived, std::shared_ptr<vrederived>>(m, "vrederived")
        .def(py::init<>());

    m.def("make_shared_cderived",
          make_shared_cderived,
          py::arg("use_custom_deleter") = false);
    m.def("make_shared_cderived_up_cast",
          make_shared_cderived_up_cast,
          py::arg("use_custom_deleter") = false);
    m.def("pass_shared_cbase", pass_shared_cbase);
    m.def("pass_shared_cderived", pass_shared_cderived);

    m.def("make_shared_vderived",
          make_shared_vderived,
          py::arg("use_custom_deleter") = false);
    m.def("make_shared_vderived_up_cast",
          make_shared_vderived_up_cast,
          py::arg("use_custom_deleter") = false);
    m.def("pass_shared_vbase", pass_shared_vbase);
    m.def("pass_shared_vderived", pass_shared_vderived);
    m.def("pass_shared_vrederived", pass_shared_vrederived);
}

}  // namespace smart_ptr_base_derived
}  // namespace pybind11_tests
