/*
    tests/test_async.cpp -- __await__ support

    Copyright (c) 2019 Google Inc.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

#define USE_MRC_AAA
#ifdef USE_MRC_AAA
namespace mrc_ns { // minimal real caster

template <typename ValType>
struct type_mrc {
    int value = -9999;
};

template <typename CType>
struct minimal_real_caster {
    static constexpr auto name = py::detail::const_name<CType>();
    static constexpr std::uint64_t universally_unique_identifier = 1000000;

    static py::handle
    cast(CType const &src, py::return_value_policy /*policy*/, py::handle /*parent*/) {
        return py::int_(src.value + 1010).release();
    }

    // Maximizing simplicity. This will go terribly wrong for other arg types.
    template <typename>
    using cast_op_type = const CType &;

    // NOLINTNEXTLINE(google-explicit-constructor)
    operator CType const &() {
        static CType obj;
        obj.value = 11;
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
template <typename ValType>
struct type_caster<mrc_ns::type_mrc<ValType>> : mrc_ns::minimal_real_caster<mrc_ns::type_mrc<ValType>> {};
} // namespace detail
} // namespace pybind11
#endif

TEST_SUBMODULE(async_module, m) {
    struct DoesNotSupportAsync {};
    py::class_<DoesNotSupportAsync>(m, "DoesNotSupportAsync").def(py::init<>());
    struct SupportsAsync {};
    py::class_<SupportsAsync>(m, "SupportsAsync")
        .def(py::init<>())
        .def("__await__", [](const SupportsAsync &self) -> py::object {
            static_cast<void>(self);
            py::object loop = py::module_::import("asyncio.events").attr("get_event_loop")();
            py::object f = loop.attr("create_future")();
            f.attr("set_result")(5);
            return f.attr("__await__")();
        });
#ifdef USE_MRC_AAA
    m.def("type_mrc_to_python", []() { return mrc_ns::type_mrc<int>{101}; });
    m.def("type_mrc_from_python", [](const mrc_ns::type_mrc<int> &obj) { return obj.value + 100; });
#endif
}
