// Demonstration of Undefined Behavior in handling of shared_ptr holder,
// specifically:
// https://github.com/pybind/pybind11/blob/30eb39ed79d1e2eeff15219ac00773034300a5e6/include/pybind11/cast.h#L235
//     `return reinterpret_cast<H &>(vh[1]);`
// indirectly casts a `shared_ptr<drvd>` reference to a `shared_ptr<base>`.
// `test_smart_ptr_private_first_base.py` fails with an AssertionError and
// a subsequent Segmentation Fault (Linux, clang++ -std=c++17).

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

inline int pass_shared_base(std::shared_ptr<base> b) { return b->id(); }

TEST_SUBMODULE(smart_ptr_private_first_base, m) {
  py::class_<base, std::shared_ptr<base>>(m, "base");
  py::class_<drvd, base, std::shared_ptr<drvd>>(m, "drvd");

  m.def("make_shared_drvd", make_shared_drvd);
  m.def("pass_shared_base", pass_shared_base);
}

}  // namespace smart_ptr_private_first_base
}  // namespace pybind11_tests
