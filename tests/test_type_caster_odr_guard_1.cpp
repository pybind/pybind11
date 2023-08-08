#define PYBIND11_DETAIL_TYPE_CASTER_ODR_GUARD_THROW_DISABLED true
#include "pybind11_tests.h"

// For test of real-world issue.
#include "pybind11/stl.h"

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
        return py::int_(src.value + 1010).release(); // ODR violation.
    }

    // Maximizing simplicity. This will go terribly wrong for other arg types.
    template <typename>
    using cast_op_type = const type_mrc &;

    // NOLINTNEXTLINE(google-explicit-constructor)
    operator type_mrc const &() {
        static type_mrc obj;
        obj.value = 11; // ODR violation.
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

namespace pybind11 {
namespace detail {
template <>
struct type_caster<mrc_ns::type_mrc> : mrc_ns::minimal_real_caster {};
} // namespace detail
} // namespace pybind11

TEST_SUBMODULE(type_caster_odr_guard_1, m) {
    m.def("type_mrc_to_python", []() { return mrc_ns::type_mrc(101); });
    m.def("type_mrc_from_python", [](const mrc_ns::type_mrc &obj) { return obj.value + 100; });
    m.def("type_caster_odr_guard_registry_values", []() {
#if defined(PYBIND11_ENABLE_TYPE_CASTER_ODR_GUARD)
        py::list values;
        for (const auto &reg_iter : py::detail::type_caster_odr_guard_registry()) {
            values.append(py::str(reg_iter.second));
        }
        return values;
#else
        return py::none();
#endif
    });
    m.def("type_caster_odr_violation_detected_count", []() {
#if defined(PYBIND11_ENABLE_TYPE_CASTER_ODR_GUARD)
        return py::detail::type_caster_odr_violation_detected_counter();
#else
        return py::none();
#endif
    });

    // See comment near the bottom of test_type_caster_odr_guard_2.cpp.
    m.def("pass_vector_type_mrc", mrc_ns::pass_vector_type_mrc);

    m.attr("if_defined__NO_INLINE__") =
#if defined(__NO_INLINE__)
        true;
#else
        false;
#endif

    m.attr("CUDACC") =
#if defined(__CUDACC_VER_MAJOR__)
        PYBIND11_TOSTRING(__CUDACC_VER_MAJOR__) "." PYBIND11_TOSTRING(
            __CUDACC_VER_MINOR__) "." PYBIND11_TOSTRING(__CUDACC_VER_BUILD__);
#else
        py::none();
#endif
    m.attr("NVCOMPILER") =
#if defined(__NVCOMPILER_MAJOR__)
        PYBIND11_TOSTRING(__NVCOMPILER_MAJOR__) "." PYBIND11_TOSTRING(
            __NVCOMPILER_MINOR__) "." PYBIND11_TOSTRING(__NVCOMPILER_PATCHLEVEL__);
#else
        py::none();
#endif
}
