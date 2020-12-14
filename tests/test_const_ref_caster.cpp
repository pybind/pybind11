/*
    tests/test_const_ref_caster.cpp

    This test verifies that const-ref is propagated through type_caster cast_op.

    The returned ConstRefCasted type is a mimimal type that is constructed to
    reference the casting mode used.
*/

#include <pybind11/complex.h>

#include "pybind11_tests.h"


struct ConstRefCasted {
  bool is_const;
  bool is_ref;
};

PYBIND11_NAMESPACE_BEGIN(pybind11)
PYBIND11_NAMESPACE_BEGIN(detail)
template <>
class type_caster<ConstRefCasted> {
 public:
  static constexpr auto name = _<ConstRefCasted>();

  bool load(handle, bool) { return true; }

  operator ConstRefCasted&&() { value = {false, false}; return std::move(value); }
  operator ConstRefCasted&() { value = {false, true}; return value; }
  operator ConstRefCasted*() { value = {false, false}; return &value; }

  operator const ConstRefCasted&() { value = {true, true}; return value; }
  operator const ConstRefCasted*() { value = {true, false}; return &value; }

  // 
  template <typename T_>
  using cast_op_type =
      /// const
      conditional_t<
          std::is_same<remove_reference_t<T_>, const ConstRefCasted*>::value, const ConstRefCasted*,
      conditional_t<
          std::is_same<T_, const ConstRefCasted&>::value, const ConstRefCasted&,
      // non-const
      conditional_t<
          std::is_same<remove_reference_t<T_>, ConstRefCasted*>::value, ConstRefCasted*,
      conditional_t<
          std::is_same<T_, ConstRefCasted&>::value, ConstRefCasted&,
          /* else */ConstRefCasted&&>>>>;

 private:
  ConstRefCasted value = {false, false};
};
PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(pybind11)

TEST_SUBMODULE(const_ref_caster, m) {
  py::class_<ConstRefCasted>(m, "ConstRefCasted",
                             "A `py::class_` type for testing")
      .def(py::init<>())
      .def_readonly("is_const", &ConstRefCasted::is_const)
      .def_readonly("is_ref", &ConstRefCasted::is_ref);


  m.def("takes", [](ConstRefCasted x) {
    return !x.is_const && !x.is_ref;
  });
  m.def("takes_ptr", [](ConstRefCasted* x) {
    return !x->is_const && !x->is_ref;
  });
  m.def("takes_ref", [](ConstRefCasted& x) {
    return !x.is_const && x.is_ref;
  });
  m.def("takes_ref_wrap", [](std::reference_wrapper<ConstRefCasted> x) {
    return !x.get().is_const && x.get().is_ref;
  });

  m.def("takes_const_ptr", [](const ConstRefCasted* x) {
    return x->is_const && !x->is_ref;
  });
  m.def("takes_const_ref", [](const ConstRefCasted& x) {
    return x.is_const && x.is_ref;
  });
  m.def("takes_const_ref_wrap",
        [](std::reference_wrapper<const ConstRefCasted> x) {
          return x.get().is_const && x.get().is_ref;
        });
}
