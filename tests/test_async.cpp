/*
    tests/test_async.cpp -- __await__ support

    Copyright (c) 2019 Google Inc.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

namespace mrc_ns { // minimal real caster

struct minimal_real_caster;

struct type_mrc {
    int value = -9999;
};

struct minimal_real_caster {
    static constexpr auto name = py::detail::const_name<type_mrc>();

    static py::handle
    cast(type_mrc const &src, py::return_value_policy /*policy*/, py::handle /*parent*/) {
        return py::int_(src.value + 1010).release();
    }

    // Maximizing simplicity. This will go terribly wrong for other arg types.
    template <typename>
    using cast_op_type = const type_mrc &;

    // NOLINTNEXTLINE(google-explicit-constructor)
    operator type_mrc const &() {
        static type_mrc obj;
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
template <>
struct type_caster<mrc_ns::type_mrc> : mrc_ns::minimal_real_caster {};
} // namespace detail
} // namespace pybind11

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
    m.def("type_mrc_to_python", []() { return mrc_ns::type_mrc{101}; });
    m.def("type_mrc_from_python", [](const mrc_ns::type_mrc &obj) { return obj.value + 100; });
}
