#include "pybind11_tests.h"

#include <pybind11/smart_holder_poc.h>

#include <iostream>
#include <string>

namespace pybind11_tests {
namespace smart_holder_poc {

inline void to_cout(std::string msg) { std::cout << msg << std::endl; }

inline void exercise() {
  to_cout("");
  namespace py = pybind11;
  py::smart_holder hld;
  hld.from_raw_ptr_owned(new int(13));
  to_cout(hld.rtti_held->name());
  {
    std::shared_ptr<int> val = hld.as_shared_ptr<int>();
    to_cout(std::to_string(*val));
  }
  {
    std::unique_ptr<int> val(hld.as_raw_ptr_owned<int>());
    to_cout(std::to_string(*val));
  }
}

TEST_SUBMODULE(smart_holder_poc, m) { m.def("exercise", exercise); }

}  // namespace smart_holder_poc
}  // namespace pybind11_tests
