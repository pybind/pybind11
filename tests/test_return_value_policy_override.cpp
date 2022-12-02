#include <pybind11/smart_holder.h>

#include "pybind11_tests.h"

namespace test_return_value_policy_override {

struct some_type {};

struct obj {
    std::string mtxt;
    obj(const std::string &mtxt_) : mtxt(mtxt_) {}
    obj(const obj &other) { mtxt = other.mtxt + "_CpCtor"; }
    obj(obj &&other) { mtxt = other.mtxt + "_MvCtor"; }
    obj &operator=(const obj &other) {
        mtxt = other.mtxt + "_CpCtor";
        return *this;
    }
    obj &operator=(obj &&other) {
        mtxt = other.mtxt + "_MvCtor";
        return *this;
    }
};

struct nocopy {
    std::string mtxt;
    nocopy(const std::string &mtxt_) : mtxt(mtxt_) {}
    nocopy(const nocopy &) = delete;
    nocopy(nocopy &&other) { mtxt = other.mtxt + "_MvCtor"; }
    nocopy &operator=(const nocopy &) = delete;
    nocopy &operator=(nocopy &&other) {
        mtxt = other.mtxt + "_MvCtor";
        return *this;
    }
};

obj *return_pointer() {
    static obj value("pointer");
    return &value;
}

const obj *return_const_pointer() {
    static obj value("const_pointer");
    return &value;
}

obj &return_reference() {
    static obj value("reference");
    return value;
}

const obj &return_const_reference() {
    static obj value("const_reference");
    return value;
}

std::shared_ptr<obj> return_shared_pointer() {
    return std::shared_ptr<obj>(new obj("shared_pointer"));
}

std::unique_ptr<obj> return_unique_pointer() {
    return std::unique_ptr<obj>(new obj("unique_pointer"));
}

nocopy &return_reference_nocopy() {
    static nocopy value("reference_nocopy");
    return value;
}

} // namespace test_return_value_policy_override

using test_return_value_policy_override::nocopy;
using test_return_value_policy_override::obj;
using test_return_value_policy_override::some_type;

namespace pybind11 {
namespace detail {

const char *return_value_policy_name(return_value_policy policy) {
    switch (policy) {
        case return_value_policy::automatic:
            return "automatic";
        case return_value_policy::automatic_reference:
            return "automatic_reference";
        case return_value_policy::take_ownership:
            return "take_ownership";
        case return_value_policy::copy:
            return "copy";
        case return_value_policy::move:
            return "move";
        case return_value_policy::reference:
            return "reference";
        case return_value_policy::reference_internal:
            return "reference_internal";
        case return_value_policy::_return_as_bytes:
            return "_return_as_bytes";
        case return_value_policy::_clif_automatic:
            return "_clif_automatic";
        default:
            return "Expected to be unreachable.";
    }
};

template <>
struct type_caster<some_type> : type_caster_base<some_type> {

    static handle cast(some_type &&, return_value_policy policy, handle /*parent*/) {
        return str(std::string(return_value_policy_name(policy))).release().ptr();
    }

    static handle cast(some_type *, return_value_policy policy, handle /*parent*/) {
        return str(std::string(return_value_policy_name(policy))).release().ptr();
    }
};

} // namespace detail
} // namespace pybind11

PYBIND11_SMART_HOLDER_TYPE_CASTERS(obj)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(nocopy)

TEST_SUBMODULE(return_value_policy_override, m) {
    m.def("return_value_with_default_policy", []() { return some_type(); });
    m.def(
        "return_value_with_policy_copy",
        []() { return some_type(); },
        py::return_value_policy::copy);
    m.def(
        "return_value_with_policy_clif_automatic",
        []() { return some_type(); },
        py::return_value_policy::_clif_automatic);
    m.def("return_pointer_with_default_policy", []() {
        static some_type value;
        return &value;
    });
    m.def(
        "return_pointer_with_policy_move",
        []() {
            static some_type value;
            return &value;
        },
        py::return_value_policy::move);
    m.def(
        "return_pointer_with_policy_clif_automatic",
        []() {
            static some_type value;
            return &value;
        },
        py::return_value_policy::_clif_automatic);

    py::classh<obj>(m, "object").def(py::init<std::string>()).def_readonly("mtxt", &obj::mtxt);
    m.def(
        "return_object_value_with_policy_clif_automatic",
        []() { return obj("value"); },
        py::return_value_policy::_clif_automatic);
    // test_return_value_policy_override::return_pointer with default policy
    // causes crash
    m.def("return_object_pointer_with_policy_clif_automatic",
          &test_return_value_policy_override::return_pointer,
          py::return_value_policy::_clif_automatic);
    // test_return_value_policy_override::return_const_pointer with default
    // policy causes crash
    m.def("return_object_const_pointer_with_policy_clif_automatic",
          &test_return_value_policy_override::return_const_pointer,
          py::return_value_policy::_clif_automatic);
    m.def("return_object_reference_with_policy_clif_automatic",
          &test_return_value_policy_override::return_reference,
          py::return_value_policy::_clif_automatic);
    m.def("return_object_const_reference_with_policy_clif_automatic",
          &test_return_value_policy_override::return_const_reference,
          py::return_value_policy::_clif_automatic);
    m.def("return_object_unique_ptr_with_policy_clif_automatic",
          &test_return_value_policy_override::return_unique_pointer,
          py::return_value_policy::_clif_automatic);
    m.def("return_object_shared_ptr_with_policy_clif_automatic",
          &test_return_value_policy_override::return_shared_pointer,
          py::return_value_policy::_clif_automatic);

    py::classh<nocopy>(m, "nocopy")
        .def(py::init<std::string>())
        .def_readonly("mtxt", &nocopy::mtxt);
    m.def("return_nocopy_reference_with_policy_clif_automatic",
          &test_return_value_policy_override::return_reference_nocopy,
          py::return_value_policy::_clif_automatic);
}
