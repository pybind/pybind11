#include "pybind11_tests.h"

#define USE_MRC_AAA
#ifdef USE_MRC_AAA
namespace mrc_ns { // minimal real caster

struct type_mrc {
    int value = -9999;
};

template <typename Ignored = void>
struct minimal_real_caster {
    static constexpr auto name = py::detail::const_name<type_mrc>();
    static std::int32_t odr_guard; // WANTED: ASAN detect_odr_violation

    static py::handle
    cast(type_mrc const &src, py::return_value_policy /*policy*/, py::handle /*parent*/) {
        odr_guard++;                                 // Just to make sure it is used.
        return py::int_(src.value + 1010).release(); // Actual ODR violation.
    }

    // Maximizing simplicity. This will go terribly wrong for other arg types.
    template <typename>
    using cast_op_type = const type_mrc &;

    // NOLINTNEXTLINE(google-explicit-constructor)
    operator type_mrc const &() {
        static type_mrc obj;
        obj.value = 11; // Actual ODR violation.
        return obj;
    }

    bool load(py::handle src, bool /*convert*/) {
        // Only accepts str, but the value is ignored.
        return py::isinstance<py::str>(src);
    }
};

template <typename Ignored>
std::int32_t minimal_real_caster<Ignored>::odr_guard = 0;

} // namespace mrc_ns

namespace pybind11 {
namespace detail {
template <>
struct type_caster<mrc_ns::type_mrc> : mrc_ns::minimal_real_caster<> {};
} // namespace detail
} // namespace pybind11
#endif

TEST_SUBMODULE(odr_guard_1, m) {
#ifdef USE_MRC_AAA
    m.def("sizeof_mrc_odr_guard",
          []() { return sizeof(mrc_ns::minimal_real_caster<>::odr_guard); });
    m.def("type_mrc_to_python", []() { return mrc_ns::type_mrc{101}; });
    m.def("type_mrc_from_python", [](const mrc_ns::type_mrc &obj) { return obj.value + 100; });
    m.def("mrc_odr_guard", []() { return mrc_ns::minimal_real_caster<>::odr_guard; });
#endif
}
