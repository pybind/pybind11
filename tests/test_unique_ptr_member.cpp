#include "pybind11_tests.h"

#include <pybind11/vptr_holder.h>

#include <iostream>
#include <memory>

namespace pybind11_tests {
namespace unique_ptr_member {

inline void to_cout(std::string text) { std::cout << text << std::endl; }

class pointee { // NOT copyable.
  public:
    pointee() = default;

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

class ptr_owner {
  public:
    explicit ptr_owner(std::unique_ptr<pointee> ptr) : ptr_(std::move(ptr)) {}

    bool is_owner() const { return bool(ptr_); }

    std::unique_ptr<pointee> give_up_ownership_via_unique_ptr() {
        return std::move(ptr_);
    }
    std::shared_ptr<pointee> give_up_ownership_via_shared_ptr() {
        return std::move(ptr_);
    }

  private:
    std::unique_ptr<pointee> ptr_;
};

// Just to have a minimal example of a typical C++ pattern.
inline int cpp_pattern() {
    auto obj = make_unique_pointee();
    int result = (obj ? 1 : 8);
    obj->get_int();
    ptr_owner owner(std::move(obj));
    result = result * 10 + (obj ? 8 : 1);
    result = result * 10 + (owner.is_owner() ? 1 : 8);
    to_cout("before give up");
    auto reclaimed = owner.give_up_ownership_via_shared_ptr();
    to_cout("after give up");
    result = result * 10 + (owner.is_owner() ? 8 : 1);
    result = result * 10 + (reclaimed ? 1 : 8);
    reclaimed.reset();
    to_cout("after del");
    result = result * 10 + (reclaimed ? 8 : 1);
    return result;
}

} // namespace unique_ptr_member
} // namespace pybind11_tests

namespace pybind11 {
namespace detail {
template <>
struct type_caster<
    std::unique_ptr<pybind11_tests::unique_ptr_member::pointee>> {
  public:
    PYBIND11_TYPE_CASTER(
        std::unique_ptr<pybind11_tests::unique_ptr_member::pointee>,
        _("std::unique_ptr<pybind11_tests::unique_ptr_member::pointee>"));

    bool load(handle /* src */, bool) {
        throw std::runtime_error("Not implemented: load");
    }

    static handle
    cast(std::unique_ptr<pybind11_tests::unique_ptr_member::pointee> /* src */,
         return_value_policy /* policy */, handle /* parent */) {
        throw std::runtime_error("Not implemented: cast");
    }
};
} // namespace detail
} // namespace pybind11

namespace pybind11_tests {
namespace unique_ptr_member {

TEST_SUBMODULE(unique_ptr_member, m) {
    m.def("to_cout", to_cout);

    py::class_<pointee, py::vptr_holder<pointee>>(m, "pointee")
        .def(py::init<>())
        .def("get_int", &pointee::get_int);

    m.def("make_unique_pointee", make_unique_pointee);

    py::class_<ptr_owner>(m, "ptr_owner")
        .def(py::init<std::unique_ptr<pointee>>(), py::arg("ptr"))
        .def("is_owner", &ptr_owner::is_owner)
        .def("give_up_ownership_via_unique_ptr",
             &ptr_owner::give_up_ownership_via_unique_ptr)
        .def("give_up_ownership_via_shared_ptr",
             &ptr_owner::give_up_ownership_via_shared_ptr);

    m.def("cpp_pattern", cpp_pattern);
}

} // namespace unique_ptr_member
} // namespace pybind11_tests
