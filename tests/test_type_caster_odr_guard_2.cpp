#define PYBIND11_DETAIL_TYPE_CASTER_ODR_GUARD_THROW_DISABLED true
#include "pybind11_tests.h"

// For test of real-world issue.
#include "pybind11/stl_bind.h"

#include <vector>

namespace mrc_ns { // minimal real caster

struct type_mrc {
    explicit type_mrc(int v = -9999) : value(v) {}
    int value;
};

struct minimal_real_caster {
    static constexpr auto name = py::detail::const_name<type_mrc>();

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

// Intentionally not called from Python: this test is to exercise the ODR guard,
// not stl.h or stl_bind.h.
inline void pass_vector_type_mrc(const std::vector<type_mrc> &) {}

} // namespace mrc_ns

PYBIND11_MAKE_OPAQUE(std::vector<mrc_ns::type_mrc>);

namespace pybind11 {
namespace detail {
template <>
struct type_caster<mrc_ns::type_mrc> : mrc_ns::minimal_real_caster {};
} // namespace detail
} // namespace pybind11

TEST_SUBMODULE(type_caster_odr_guard_2, m) {
    m.def("type_mrc_to_python", []() { return mrc_ns::type_mrc(202); });
    m.def("type_mrc_from_python", [](const mrc_ns::type_mrc &obj) { return obj.value + 200; });

    // Uncomment and run test_type_caster_odr_guard_1.py to verify that the
    // test_type_caster_odr_violation_detected_counter subtest fails
    // (num_violations 2 instead of 1).
    // Unlike the "controlled ODR violation" for the minimal_real_caster, this ODR violation is
    // completely unsafe, therefore it cannot portably be exercised with predictable results.
    // m.def("pass_vector_type_mrc", mrc_ns::pass_vector_type_mrc);
}
