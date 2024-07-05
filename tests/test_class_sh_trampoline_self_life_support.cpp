// Copyright (c) 2021 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pybind11/smart_holder.h"
#include "pybind11/trampoline_self_life_support.h"
#include "pybind11_tests.h"

#include <memory>
#include <string>
#include <utility>

namespace {

struct Big5 { // Also known as "rule of five".
    std::string history;

    explicit Big5(std::string history_start) : history{std::move(history_start)} {}

    Big5(const Big5 &other) { history = other.history + "_CpCtor"; }

    Big5(Big5 &&other) noexcept { history = other.history + "_MvCtor"; }

    Big5 &operator=(const Big5 &other) {
        history = other.history + "_OpEqLv";
        return *this;
    }

    Big5 &operator=(Big5 &&other) noexcept {
        history = other.history + "_OpEqRv";
        return *this;
    }

    virtual ~Big5() = default;

protected:
    Big5() : history{"DefaultConstructor"} {}
};

struct Big5Trampoline : Big5, py::trampoline_self_life_support {
    using Big5::Big5;
};

} // namespace

PYBIND11_SMART_HOLDER_TYPE_CASTERS(Big5)

TEST_SUBMODULE(class_sh_trampoline_self_life_support, m) {
    py::classh<Big5, Big5Trampoline>(m, "Big5")
        .def(py::init<std::string>())
        .def_readonly("history", &Big5::history);

    m.def("action", [](std::unique_ptr<Big5> obj, int action_id) {
        py::object o2 = py::none();
        // This is very unusual, but needed to directly exercise the trampoline_self_life_support
        // CpCtor, MvCtor, operator= lvalue, operator= rvalue.
        auto *obj_trampoline = dynamic_cast<Big5Trampoline *>(obj.get());
        if (obj_trampoline != nullptr) {
            switch (action_id) {
                case 0: { // CpCtor
                    std::unique_ptr<Big5> cp(new Big5Trampoline(*obj_trampoline));
                    o2 = py::cast(std::move(cp));
                } break;
                case 1: { // MvCtor
                    std::unique_ptr<Big5> mv(new Big5Trampoline(std::move(*obj_trampoline)));
                    o2 = py::cast(std::move(mv));
                } break;
                case 2: { // operator= lvalue
                    std::unique_ptr<Big5> lv(new Big5Trampoline);
                    *lv = *obj_trampoline; // NOLINT clang-tidy cppcoreguidelines-slicing
                    o2 = py::cast(std::move(lv));
                } break;
                case 3: { // operator= rvalue
                    std::unique_ptr<Big5> rv(new Big5Trampoline);
                    *rv = std::move(*obj_trampoline);
                    o2 = py::cast(std::move(rv));
                } break;
                default:
                    break;
            }
        }
        py::object o1 = py::cast(std::move(obj));
        return py::make_tuple(o1, o2);
    });
}
