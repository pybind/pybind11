#include "pybind11_tests.h"

#include <memory>

namespace pybind11_tests {
namespace unique_ptr_member {

class pointee { // NOT copyable.
  public:
    pointee() = default;

    int get_int() const { return 213; }

  private:
    pointee(const pointee &) = delete;
    pointee(pointee &&) = delete;
    pointee &operator=(const pointee &) = delete;
    pointee &operator=(pointee &&) = delete;
};

class ptr_owner {
  public:
    explicit ptr_owner(std::unique_ptr<pointee> ptr) : ptr_(std::move(ptr)) {}

  private:
    std::unique_ptr<pointee> ptr_;
};

// Just to have a minimal example of a typical C++ pattern.
inline int cpp_pattern() {
    auto obj = std::unique_ptr<pointee>(new pointee);
    int result = (obj ? 10 : 0);
    ptr_owner owner(std::move(obj));
    result += (obj ? 1 : 0);
    return result;
}

TEST_SUBMODULE(unique_ptr_member, m) {
    m.def("cpp_pattern", cpp_pattern);

    py::class_<pointee>(m, "pointee")
        .def(py::init<>())
        .def("get_int", &pointee::get_int);

    py::class_<ptr_owner>(m, "ptr_owner")
#ifdef FEAT_UNIQUE_PTR_ARG
        .def(py::init<std::unique_ptr<pointee>>(), py::arg("ptr"))
#endif
        ;
}

} // namespace unique_ptr_member
} // namespace pybind11_tests
