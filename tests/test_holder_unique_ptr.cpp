// KEEP IN SYNC WITH test_holder_shared_ptr.cpp

#include "pybind11_tests.h"

#include <iostream>
#include <memory>

namespace pybind11_tests {
namespace holder_unique_ptr {

inline void to_cout(std::string text) { std::cout << text << std::endl; }

class pointee { // NOT copyable.
  public:
    pointee() { to_cout("pointee::pointee()"); }

    int get_int() const {
        to_cout("pointee::get_int()");
        return 213;
    }

    ~pointee() { to_cout("~pointee()"); }

  private:
    pointee(const pointee &) = delete;
    pointee(pointee &&) = delete;
    pointee &operator=(const pointee &) = delete;
    pointee &operator=(pointee &&) = delete;
};

inline std::unique_ptr<pointee> make_unique_pointee() {
    return std::unique_ptr<pointee>(new pointee);
}

inline std::shared_ptr<pointee> make_shared_pointee() {
    return std::unique_ptr<pointee>(new pointee);
}

inline int pass_unique_pointee(std::unique_ptr<pointee> ptr) {
    return 4000 + ptr->get_int();
}

inline int pass_shared_pointee(std::shared_ptr<pointee> ptr) {
    return 5000 + ptr->get_int();
}

inline pointee* get_static_pointee() {
  static pointee cpp_instance;
  return &cpp_instance;
}

TEST_SUBMODULE(holder_unique_ptr, m) {
    m.def("to_cout", to_cout);

    py::class_<pointee>(m, "pointee")
        .def(py::init<>())
        .def("get_int", &pointee::get_int);

    m.def("make_unique_pointee", make_unique_pointee);
    m.def("make_shared_pointee", make_shared_pointee);
    // m.def("pass_unique_pointee", pass_unique_pointee);
    m.def("pass_shared_pointee", pass_shared_pointee);

    m.def("get_static_pointee",
          get_static_pointee, py::return_value_policy::reference);
}

} // namespace holder_unique_ptr
} // namespace pybind11_tests
