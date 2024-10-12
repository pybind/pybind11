#include "pybind11_tests.h"

namespace test_type_caster_incomplete_type {

struct ForwardDeclaredType {};

} // namespace test_type_caster_incomplete_type

using ForwardDeclaredType = test_type_caster_incomplete_type::ForwardDeclaredType;

// TODO: Move to pybind11/type_caster_incomplete_type.h, wrap in a macro.
namespace pybind11 {
namespace detail {

template <>
class type_caster<ForwardDeclaredType> {
public:
    static constexpr auto name = const_name("object");

    static handle cast(ForwardDeclaredType * /*src*/,
                       return_value_policy /*policy*/,
                       handle /*parent*/) {
        return py::none().release(); // TODO: Build and return capsule with src pointer;
    }

    bool load(handle /*src*/, bool /*convert*/) {
        // TODO: Assign pointer_capsule = src after inspecting src.
        return true;
    }

    template <typename T>
    using cast_op_type = ForwardDeclaredType *;

    explicit operator ForwardDeclaredType *() {
        return nullptr; // TODO: Retrieve C++ pointer from pointer_capsule.
    }

private:
    capsule pointer_capsule;
};

} // namespace detail
} // namespace pybind11

TEST_SUBMODULE(type_caster_incomplete_type, m) {
    m.def("rtrn_fwd_decl_type_ptr",
          []() { return reinterpret_cast<ForwardDeclaredType *>(0); });
    m.def("pass_fwd_decl_type_ptr", [](ForwardDeclaredType *) {});
}
