// Copyright (c) 2021 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pybind11/smart_holder.h"
#include "pybind11_tests.h"

#include <memory>
#include <string>

namespace {

struct Sft : std::enable_shared_from_this<Sft> {
    std::string history;
    explicit Sft(const std::string &history) : history{history} {}
    long use_count() const {
#if defined(__cpp_lib_enable_shared_from_this) && (!defined(_MSC_VER) || _MSC_VER >= 1912)
        return this->shared_from_this().use_count();
#else
        return -1;
#endif
    }
    virtual ~Sft() = default;

#if defined(__clang__)
    // "Group of 4" begin.
    // This group is not meant to be used, but will leave a trace in the
    // history in case something goes wrong.
    // However, compilers other than clang have a variety of issues. It is not
    // worth the trouble covering all platforms.
    Sft(const Sft &other) { history = other.history + "_CpCtor"; }

    Sft(Sft &&other) { history = other.history + "_MvCtor"; }

    Sft &operator=(const Sft &other) {
        history = other.history + "_OpEqLv";
        return *this;
    }

    Sft &operator=(Sft &&other) {
        history = other.history + "_OpEqRv";
        return *this;
    }
    // "Group of 4" end.
#endif
};

struct SftSharedPtrStash {
    int ser_no;
    std::vector<std::shared_ptr<Sft>> stash;
    explicit SftSharedPtrStash(int ser_no) : ser_no{ser_no} {}
    void Add(const std::shared_ptr<Sft> &obj) {
        obj->history += "_Stash" + std::to_string(ser_no) + "Add";
        stash.push_back(obj);
    }
    void AddSharedFromThis(Sft *obj) {
        auto sft = obj->shared_from_this();
        sft->history += "_Stash" + std::to_string(ser_no) + "AddSharedFromThis";
        stash.push_back(sft);
    }
    std::string history(unsigned i) {
        if (i < stash.size())
            return stash[i]->history;
        return "OutOfRange";
    }
    long use_count(unsigned i) {
        if (i < stash.size())
            return stash[i].use_count();
        return -1;
    }
};

struct SftTrampoline : Sft, py::trampoline_self_life_support {
    using Sft::Sft;
};

void pass_shared_ptr(const std::shared_ptr<Sft> &obj) {
    obj->shared_from_this()->history += "_PassSharedPtr";
}

void pass_unique_ptr(const std::unique_ptr<Sft> &) {}

} // namespace

PYBIND11_SMART_HOLDER_TYPE_CASTERS(Sft)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(SftSharedPtrStash)

TEST_SUBMODULE(class_sh_trampoline_shared_from_this, m) {
    py::classh<Sft, SftTrampoline>(m, "Sft")
        .def(py::init<std::string>())
        .def_readonly("history", &Sft::history)
        .def("use_count", &Sft::use_count);

    py::classh<SftSharedPtrStash>(m, "SftSharedPtrStash")
        .def(py::init<int>())
        .def("Add", &SftSharedPtrStash::Add)
        .def("AddSharedFromThis", &SftSharedPtrStash::AddSharedFromThis)
        .def("history", &SftSharedPtrStash::history)
        .def("use_count", &SftSharedPtrStash::use_count);

    m.def("pass_shared_ptr", pass_shared_ptr);
    m.def("pass_unique_ptr", pass_unique_ptr);
}
