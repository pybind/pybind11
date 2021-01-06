#include "pybind11_tests.h"

#include <pybind11/smart_holder_poc.h>

#include <iostream>
#include <string>

namespace pybind11_tests {
namespace smart_holder_poc {

inline void to_cout(std::string msg) { std::cout << msg << std::endl; }

template <typename T>
struct functor_builtin_delete {
  void operator()(T* ptr) { delete ptr; }
};

inline void exercise() {
  to_cout("");
  namespace py = pybind11;
  {
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
  }  // namespace ;
  {
    std::unique_ptr<int> val(new int(13));
    py::smart_holder hld;
    hld.from_raw_ptr_unowned(val.get());
    to_cout(std::to_string(*hld.as_raw_ptr_unowned<int>()));
  }
  {
    std::unique_ptr<int> val(new int(13));
    py::smart_holder hld;
    hld.from_unique_ptr(std::move(val));
    to_cout(std::to_string(*hld.as_raw_ptr_unowned<int>()));
  }
  {
    py::smart_holder hld;
    hld.from_raw_ptr_owned(new int(13));
    to_cout(std::to_string(*hld.as_unique_ptr<int>()));
  }
  {
    std::unique_ptr<int, functor_builtin_delete<int>> unq_ptr(new int(13));
    py::smart_holder hld;
    hld.from_unique_ptr_with_deleter(std::move(unq_ptr));
    to_cout(std::to_string(unq_ptr.get() == nullptr));
    to_cout(std::to_string(*hld.as_raw_ptr_unowned<int>()));
    auto unq_ptr_retrieved =
        hld.as_unique_ptr_with_deleter<int, functor_builtin_delete<int>>();
    to_cout(std::to_string(hld.vptr.get() == nullptr));
    to_cout(std::to_string(*unq_ptr_retrieved));
  }
}

TEST_SUBMODULE(smart_holder_poc, m) { m.def("exercise", exercise); }

}  // namespace smart_holder_poc
}  // namespace pybind11_tests
