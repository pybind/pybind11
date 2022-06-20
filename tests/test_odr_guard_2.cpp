#include "pybind11_tests.h"

namespace mrc_ns { // minimal real caster

struct type_mrc {
    int value = -9999;
};

struct minimal_real_caster {
    static constexpr auto name = py::detail::const_name<type_mrc>();
    static constexpr std::uint64_t universally_unique_identifier = 2000;

    static py::handle
    cast(type_mrc const &src, py::return_value_policy /*policy*/, py::handle /*parent*/) {
        return py::int_(src.value + 2020).release(); // ODR violation.
    }

    // Maximizing simplicity. This will go terribly wrong for other arg types.
    template <typename>
    using cast_op_type = const type_mrc &;

    // NOLINTNEXTLINE(google-explicit-constructor)
    operator type_mrc const &() {
        static type_mrc obj;
        obj.value = 22; // ODR violation.
        return obj;
    }

    bool load(py::handle src, bool /*convert*/) {
        // Only accepts str, but the value is ignored.
        return py::isinstance<py::str>(src);
    }
};

} // namespace mrc_ns

namespace pybind11 {
namespace detail {
template <>
struct type_caster<mrc_ns::type_mrc> : mrc_ns::minimal_real_caster {};
} // namespace detail
} // namespace pybind11

TEST_SUBMODULE(odr_guard_2, m) {
    m.def("type_mrc_to_python", []() { return mrc_ns::type_mrc{202}; });
    m.def("type_mrc_from_python", [](const mrc_ns::type_mrc &obj) { return obj.value + 200; });
}
