#include <pybind11/smart_holder.h>

#include "pybind11_tests.h"

#include <memory>

namespace pybind11_tests {
namespace class_sh_unique_ptr_member {

class pointee { // NOT copyable.
public:
    pointee() = default;

    int get_int() const { return 213; }

    pointee(const pointee &) = delete;
    pointee(pointee &&) = delete;
    pointee &operator=(const pointee &) = delete;
    pointee &operator=(pointee &&) = delete;
};

inline std::unique_ptr<pointee> make_unique_pointee() {
    return std::unique_ptr<pointee>(new pointee);
}

class ptr_owner {
public:
    explicit ptr_owner(std::unique_ptr<pointee> ptr) : ptr_(std::move(ptr)) {}

    bool is_owner() const { return bool(ptr_); }

    std::unique_ptr<pointee> give_up_ownership_via_unique_ptr() { return std::move(ptr_); }
    std::shared_ptr<pointee> give_up_ownership_via_shared_ptr() { return std::move(ptr_); }

private:
    std::unique_ptr<pointee> ptr_;
};

} // namespace class_sh_unique_ptr_member
} // namespace pybind11_tests

PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_unique_ptr_member::pointee)

namespace pybind11_tests {
namespace class_sh_unique_ptr_member {

TEST_SUBMODULE(class_sh_unique_ptr_member, m) {
    py::classh<pointee>(m, "pointee").def(py::init<>()).def("get_int", &pointee::get_int);

    m.def("make_unique_pointee", make_unique_pointee);

    py::class_<ptr_owner>(m, "ptr_owner")
        .def(py::init<std::unique_ptr<pointee>>(), py::arg("ptr"))
        .def("is_owner", &ptr_owner::is_owner)
        .def("give_up_ownership_via_unique_ptr", &ptr_owner::give_up_ownership_via_unique_ptr)
        .def("give_up_ownership_via_shared_ptr", &ptr_owner::give_up_ownership_via_shared_ptr);
}

} // namespace class_sh_unique_ptr_member
} // namespace pybind11_tests
