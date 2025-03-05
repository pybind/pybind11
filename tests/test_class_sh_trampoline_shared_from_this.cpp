// Copyright (c) 2021 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pybind11_tests.h"

#include <memory>
#include <string>

namespace pybind11_tests {
namespace class_sh_trampoline_shared_from_this {

struct Sft : std::enable_shared_from_this<Sft> {
    std::string history;
    explicit Sft(const std::string &history_seed) : history{history_seed} {}
    virtual ~Sft() = default;

#if defined(__clang__)
    // "Group of 4" begin.
    // This group is not meant to be used, but will leave a trace in the
    // history in case something goes wrong.
    // However, compilers other than clang have a variety of issues. It is not
    // worth the trouble covering all platforms.
    Sft(const Sft &other) : enable_shared_from_this(other) { history = other.history + "_CpCtor"; }

    Sft(Sft &&other) noexcept { history = other.history + "_MvCtor"; }

    Sft &operator=(const Sft &other) {
        history = other.history + "_OpEqLv";
        return *this;
    }

    Sft &operator=(Sft &&other) noexcept {
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
    void Clear() { stash.clear(); }
    void Add(const std::shared_ptr<Sft> &obj) {
        if (!obj->history.empty()) {
            obj->history += "_Stash" + std::to_string(ser_no) + "Add";
        }
        stash.push_back(obj);
    }
    void AddSharedFromThis(Sft *obj) {
        auto sft = obj->shared_from_this();
        if (!sft->history.empty()) {
            sft->history += "_Stash" + std::to_string(ser_no) + "AddSharedFromThis";
        }
        stash.push_back(sft);
    }
    std::string history(unsigned i) {
        if (i < stash.size()) {
            return stash[i]->history;
        }
        return "OutOfRange";
    }
    long use_count(unsigned i) {
        if (i < stash.size()) {
            return stash[i].use_count();
        }
        return -1;
    }
};

struct SftTrampoline : Sft, py::trampoline_self_life_support {
    using Sft::Sft;
};

long use_count(const std::shared_ptr<Sft> &obj) { return obj.use_count(); }

long pass_shared_ptr(const std::shared_ptr<Sft> &obj) {
    auto sft = obj->shared_from_this();
    if (!sft->history.empty()) {
        sft->history += "_PassSharedPtr";
    }
    return sft.use_count();
}

std::string pass_unique_ptr_cref(const std::unique_ptr<Sft> &obj) {
    return obj ? obj->history : "<NULLPTR>";
}
void pass_unique_ptr_rref(std::unique_ptr<Sft> &&) {
    throw std::runtime_error("Expected to not be reached.");
}

Sft *make_pure_cpp_sft_raw_ptr(const std::string &history_seed) { return new Sft{history_seed}; }

std::unique_ptr<Sft> make_pure_cpp_sft_unq_ptr(const std::string &history_seed) {
    return std::unique_ptr<Sft>(new Sft{history_seed});
}

std::shared_ptr<Sft> make_pure_cpp_sft_shd_ptr(const std::string &history_seed) {
    return std::make_shared<Sft>(history_seed);
}

std::shared_ptr<Sft> pass_through_shd_ptr(const std::shared_ptr<Sft> &obj) { return obj; }

} // namespace class_sh_trampoline_shared_from_this
} // namespace pybind11_tests

using namespace pybind11_tests::class_sh_trampoline_shared_from_this;

TEST_SUBMODULE(class_sh_trampoline_shared_from_this, m) {
    py::classh<Sft, SftTrampoline>(m, "Sft")
        .def(py::init<const std::string &>())
        .def(py::init([](const std::string &history, int) {
            return std::make_shared<SftTrampoline>(history);
        }))
        .def_readonly("history", &Sft::history)
        // This leads to multiple entries in registered_instances:
        .def(py::init([](const std::shared_ptr<Sft> &existing) { return existing; }));

    py::classh<SftSharedPtrStash>(m, "SftSharedPtrStash")
        .def(py::init<int>())
        .def("Clear", &SftSharedPtrStash::Clear)
        .def("Add", &SftSharedPtrStash::Add)
        .def("AddSharedFromThis", &SftSharedPtrStash::AddSharedFromThis)
        .def("history", &SftSharedPtrStash::history)
        .def("use_count", &SftSharedPtrStash::use_count);

    m.def("use_count", use_count);
    m.def("pass_shared_ptr", pass_shared_ptr);
    m.def("pass_unique_ptr_cref", pass_unique_ptr_cref);
    m.def("pass_unique_ptr_rref", pass_unique_ptr_rref);
    m.def("make_pure_cpp_sft_raw_ptr", make_pure_cpp_sft_raw_ptr);
    m.def("make_pure_cpp_sft_unq_ptr", make_pure_cpp_sft_unq_ptr);
    m.def("make_pure_cpp_sft_shd_ptr", make_pure_cpp_sft_shd_ptr);
    m.def("pass_through_shd_ptr", pass_through_shd_ptr);
}
