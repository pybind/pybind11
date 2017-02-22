/*
  tests/test_abstract_base_class.cpp

  Previously, failed to compile under debug mode.
 */

#include "pybind11_tests.h"

#include <pybind11/functional.h>

class AbstractBase {
public:
  virtual unsigned int num_nodes() = 0;
};

void func_accepting_func_accepting_base(std::function<double(AbstractBase&)> func) { }

test_initializer abstract_base_class([](py::module &m) {
    m.def("func_accepting_func_accepting_base",
          func_accepting_func_accepting_base);
});
