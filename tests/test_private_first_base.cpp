// Demonstration of UB (Undefined Behavior) in handling of polymorphic pointers,
// specifically:
// https://github.com/pybind/pybind11/blob/30eb39ed79d1e2eeff15219ac00773034300a5e6/include/pybind11/cast.h#L229
//     `return reinterpret_cast<V *&>(vh[0]);`
// casts a `void` pointer to a `base`. The `void` pointer is obtained through
// a `dynamic_cast` here:
// https://github.com/pybind/pybind11/blob/30eb39ed79d1e2eeff15219ac00773034300a5e6/include/pybind11/cast.h#L852
//     `return dynamic_cast<const void*>(src);`
// The `dynamic_cast` is well-defined:
// https://en.cppreference.com/w/cpp/language/dynamic_cast
//     4) If expression is a pointer to a polymorphic type, and new-type
//        is a pointer to void, the result is a pointer to the most derived
//        object pointed or referenced by expression.
// But the `reinterpret_cast` above is UB: `test_make_drvd_pass_base` in
// `test_private_first_base.py` fails with a Segmentation Fault (Linux,
// clang++ -std=c++17).
// The only well-defined cast is back to a `drvd` pointer (`static_cast` can be
// used), which can then safely be cast up to a `base` pointer. Note that
// `test_make_drvd_up_cast_pass_drvd` passes because the `void` pointer is cast
// to `drvd` pointer in this situation.

#include "pybind11_tests.h"

namespace pybind11_tests {
namespace private_first_base {

struct base {
  base() : base_id(100) {}
  virtual ~base() = default;
  virtual int id() const { return base_id; }
  base(const base&) = default;
  int base_id;
};

struct private_first_base {  // Any class with a virtual function will do.
  private_first_base() {}
  virtual void some_other_virtual_function() const {}
  virtual ~private_first_base() = default;
  private_first_base(const private_first_base&) = default;
};

struct drvd : private private_first_base, public base {
  drvd() {}
  int id() const override { return 2 * base_id; }
};

inline drvd* make_drvd() { return new drvd; }
inline base* make_drvd_up_cast() { return new drvd; }

inline int pass_base(const base* b) { return b->id(); }
inline int pass_drvd(const drvd* d) { return d->id(); }

TEST_SUBMODULE(private_first_base, m) {
  py::class_<base>(m, "base");
  py::class_<drvd, base>(m, "drvd");

  m.def("make_drvd", make_drvd,
        py::return_value_policy::take_ownership);
  m.def("make_drvd_up_cast", make_drvd_up_cast,
        py::return_value_policy::take_ownership);
  m.def("pass_base", pass_base);
  m.def("pass_drvd", pass_drvd);
}

}  // namespace private_first_base
}  // namespace pybind11_tests
