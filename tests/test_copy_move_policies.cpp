/*
    tests/test_copy_move_policies.cpp -- 'copy' and 'move'
                                         return value policies

    Copyright (c) 2016 Ben North <ben@redfrontdoor.org>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

template <typename derived>
struct empty {
    static const derived& get_one() { return instance_; }
    static derived instance_;
};

struct lacking_copy_ctor : public empty<lacking_copy_ctor> {
    lacking_copy_ctor() {}
    lacking_copy_ctor(const lacking_copy_ctor& other) = delete;
};

template <> lacking_copy_ctor empty<lacking_copy_ctor>::instance_ = {};

struct lacking_move_ctor : public empty<lacking_move_ctor> {
    lacking_move_ctor() {}
    lacking_move_ctor(const lacking_move_ctor& other) = delete;
    lacking_move_ctor(lacking_move_ctor&& other) = delete;
};

template <> lacking_move_ctor empty<lacking_move_ctor>::instance_ = {};

test_initializer copy_move_policies([](py::module &m) {
    py::class_<lacking_copy_ctor>(m, "lacking_copy_ctor")
        .def_static("get_one", &lacking_copy_ctor::get_one,
                    py::return_value_policy::copy);
    py::class_<lacking_move_ctor>(m, "lacking_move_ctor")
        .def_static("get_one", &lacking_move_ctor::get_one,
                    py::return_value_policy::move);
});
