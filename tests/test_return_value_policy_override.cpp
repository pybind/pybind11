#include "pybind11_tests.h"

namespace test_return_value_policy_override {

struct some_type {};

} // namespace test_return_value_policy_override

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
}
