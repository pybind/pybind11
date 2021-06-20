// Copyright (c) 2021 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pybind11/smart_holder.h"
#include "pybind11_tests.h"

#include <memory>
#include <string>

namespace {

struct WithSft : std::enable_shared_from_this<WithSft> {
    virtual ~WithSft() = default;
};

struct WithSftTrampoline : WithSft {
    using WithSft::WithSft;
};

void pass_shared_ptr(const std::shared_ptr<WithSft> &obj) {
    to_cout("LOOOK pass_shared_ptr entry");
    to_cout("LOOOK obj->shared_from_this();");
    (void) obj->shared_from_this();
    to_cout("LOOOK pass_shared_ptr return");
}

} // namespace

PYBIND11_SMART_HOLDER_TYPE_CASTERS(WithSft)

TEST_SUBMODULE(class_sh_trampoline_shared_from_this, m) {
    py::classh<WithSft, WithSftTrampoline>(m, "WithSft").def(py::init<>());
    m.def("pass_shared_ptr", pass_shared_ptr);
    m.def("to_cout", to_cout);
}
