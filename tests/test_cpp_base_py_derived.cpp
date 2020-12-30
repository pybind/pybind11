// pybind11 equivalent of Boost.Python test:
// https://github.com/rwgk/rwgk_tbx/blob/6c9a6d6bc72d5c1b8609724433259c5b47178680/cpp_base_py_derived_ext.cpp
// See also: https://github.com/pybind/pybind11/issues/1333 (this was the starting point)

#include "pybind11_tests.h"

namespace pybind11_tests {
namespace cpp_base_py_derived {

struct base {
  base() : base_num(100) {}

  virtual int get_num() const { return base_num; }

  virtual std::shared_ptr<base> clone() const {
    return std::shared_ptr<base>(new base(150));
  }

  virtual ~base() = default;

 private:
  explicit base(int num) : base_num(num) {}
  int base_num;
};

inline int get_num(std::shared_ptr<base> b) { return b->get_num(); }

inline int clone_get_num(std::shared_ptr<base> b) {
  std::shared_ptr<base> c = b->clone();
  return (b->get_num() + 3) * 1000 + (c->get_num() + 7);
}

struct base_trampoline : public base {
  using base::base;

  int get_num() const override {
    PYBIND11_OVERRIDE(int, base, get_num);
  }

  std::shared_ptr<base> clone() const override {
    PYBIND11_OVERRIDE(std::shared_ptr<base>, base, clone);
  }
};

TEST_SUBMODULE(cpp_base_py_derived, m) {
  py::class_<base, base_trampoline, std::shared_ptr<base>>(m, "base")
    .def(py::init<>())
    .def("get_num", &base::get_num)
    .def("clone", &base::clone)
  ;

  m.def("get_num", get_num);
  m.def("clone_get_num", clone_get_num);
}

}  // namespace cpp_base_py_derived
}  // namespace pybind11_tests
