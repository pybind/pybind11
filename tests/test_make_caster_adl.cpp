#include "pybind11_tests.h"

// adl = Argument Dependent Lookup

namespace adl_mock {
struct type_mock {};
struct mock_caster {
    static int num() { return 101; }
};
mock_caster pybind11_select_caster(type_mock *);
} // namespace adl_mock

namespace adl_mrc { // minimal real caster

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
        obj.value = 303;
        return obj;
    }

    bool load(py::handle src, bool /*convert*/) {
        // Only accepts str, but the value is ignored.
        return py::isinstance<py::str>(src);
    }
};

minimal_real_caster pybind11_select_caster(type_mrc *);

} // namespace adl_mrc

TEST_SUBMODULE(make_caster_adl, m) {
    m.def("num_mock", []() { return py::detail::make_caster<adl_mock::type_mock>::num(); });

    m.def("obj_mrc_return", []() {
        adl_mrc::type_mrc obj;
        obj.value = 404;
        return obj;
    });
    m.def("obj_mrc_arg", [](adl_mrc::type_mrc const &obj) { return obj.value + 2000; });
}
