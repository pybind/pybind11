// adl = Argument Dependent Lookup

#include "pybind11_tests.h"

namespace have_a_ns {
struct type_mock {};
struct mock_caster {
    static int num() { return 101; }
};
mock_caster pybind11_select_caster(type_mock *);
} // namespace have_a_ns

// namespace global {
struct global_ns_type_mock {};
struct global_ns_mock_caster {
    static int num() { return 202; }
};
global_ns_mock_caster pybind11_select_caster(global_ns_type_mock *);
// } // namespace global

namespace {
struct unnamed_ns_type_mock {};
struct unnamed_ns_mock_caster {
    static int num() { return 303; }
};
#if defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wunneeded-internal-declaration"
#endif
unnamed_ns_mock_caster pybind11_select_caster(unnamed_ns_type_mock *);
#if defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif
} // namespace

namespace mrc_ns { // minimal real caster

struct type_mrc {
    int value = -9999;
};

struct minimal_real_caster {
    static constexpr auto name = py::detail::const_name<type_mrc>();

    static py::handle
    cast(type_mrc const &src, py::return_value_policy /*policy*/, py::handle /*parent*/) {
        return py::int_(src.value + 1000).release();
    }

    // Maximizing simplicity. This will go terribly wrong for other arg types.
    template <typename>
    using cast_op_type = const type_mrc &;

    // NOLINTNEXTLINE(google-explicit-constructor)
    operator type_mrc const &() {
        static type_mrc obj;
        obj.value = 404;
        return obj;
    }

    bool load(py::handle src, bool /*convert*/) {
        // Only accepts str, but the value is ignored.
        return py::isinstance<py::str>(src);
    }
};

minimal_real_caster pybind11_select_caster(type_mrc *);

} // namespace mrc_ns

TEST_SUBMODULE(make_caster_adl, m) {
    m.def("have_a_ns_num", []() { return py::detail::make_caster<have_a_ns::type_mock>::num(); });
    m.def("global_ns_num", []() { return py::detail::make_caster<global_ns_type_mock>::num(); });
    m.def("unnamed_ns_num", []() { return py::detail::make_caster<unnamed_ns_type_mock>::num(); });

    m.def("mrc_return", []() {
        mrc_ns::type_mrc obj;
        obj.value = 505;
        return obj;
    });
    m.def("mrc_arg", [](mrc_ns::type_mrc const &obj) { return obj.value + 2000; });
}
