// Copyright (c) 2022 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pybind11_tests.h"

namespace have_a_ns {
struct mock_caster_alt {
    static int num() { return 121; }
};
struct type_mock {
    friend mock_caster_alt pybind11_select_caster(type_mock *);
};

} // namespace have_a_ns

TEST_SUBMODULE(select_caster_alt, m) {
    m.def("have_a_ns_num", []() { return py::detail::make_caster<have_a_ns::type_mock>::num(); });
}
