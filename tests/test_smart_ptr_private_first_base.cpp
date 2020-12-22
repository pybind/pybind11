// Demonstration of Undefined Behavior in handling of shared_ptr holder,
// specifically:
// https://github.com/pybind/pybind11/blob/30eb39ed79d1e2eeff15219ac00773034300a5e6/include/pybind11/cast.h#L235
//     `return reinterpret_cast<H &>(vh[1]);`
// indirectly casts a `shared_ptr<drvd>` reference to a `shared_ptr<base>`.
// Similarly:
// https://github.com/pybind/pybind11/blob/30eb39ed79d1e2eeff15219ac00773034300a5e6/include/pybind11/pybind11.h#L1505
//     `init_holder(inst, v_h, (const holder_type *) holder_ptr, v_h.value_ptr<type>());`
// explictly casts a `shared_ptr<base>` reference to a `shared_ptr<drvd>`.
// Both tests in `test_smart_ptr_private_first_base.py` fail with a
// Segmentation Fault (Linux, clang++ -std=c++17).

#include <memory>

#include "pybind11_tests.h"

namespace pybind11_tests {
namespace smart_ptr_private_first_base {

struct base {
  base() : base_id(100) {}
  virtual ~base() = default;
  virtual int id() const { return base_id; }
  int base_id;
};

struct private_first_base {  // Any class with a virtual function will do.
  virtual void some_other_virtual_function() const {}
  virtual ~private_first_base() = default;
};

struct drvd : private private_first_base, public base {
  int id() const override { return 2 * base_id; }
};

inline std::shared_ptr<drvd> make_shared_drvd() {
  return std::shared_ptr<drvd>(new drvd);
}

inline std::shared_ptr<base> make_shared_drvd_up_cast() {
  return std::shared_ptr<base>(new drvd);
}

inline int pass_shared_base(std::shared_ptr<base> b) { return b->id(); }
inline int pass_shared_drvd(std::shared_ptr<drvd> d) { return d->id(); }

TEST_SUBMODULE(smart_ptr_private_first_base, m) {
  py::class_<base, std::shared_ptr<base>>(m, "base");
  py::class_<drvd, base, std::shared_ptr<drvd>>(m, "drvd");

  m.def("make_shared_drvd", make_shared_drvd);
  m.def("make_shared_drvd_up_cast", make_shared_drvd_up_cast);
  m.def("pass_shared_base", pass_shared_base);
  m.def("pass_shared_drvd", pass_shared_drvd);
}

}  // namespace smart_ptr_private_first_base
}  // namespace pybind11_tests
