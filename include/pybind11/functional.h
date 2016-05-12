/*
    pybind11/functional.h: std::function<> support

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "pybind11.h"
#include <functional>

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

template <typename Return, typename... Args> struct type_caster<std::function<Return(Args...)>> {
    typedef std::function<Return(Args...)> type;
    typedef typename std::conditional<std::is_same<Return, void>::value, void_type, Return>::type retval_type;
public:
    bool load(handle src_, bool) {
        src_ = detail::get_function(src_);
        if (!src_ || !PyCallable_Check(src_.ptr()))
            return false;
        object src(src_, true);
        value = [src](Args... args) -> Return {
            gil_scoped_acquire acq;
            object retval(src(std::move(args)...));
            /* Visual studio 2015 parser issue: need parentheses around this expression */
            return (retval.template cast<Return>());
        };
        return true;
    }

    template <typename Func>
    static handle cast(Func &&f_, return_value_policy policy, handle /* parent */) {
        return cpp_function(std::forward<Func>(f_), policy).release();
    }

    PYBIND11_TYPE_CASTER(type, _("function<") +
            type_caster<std::tuple<Args...>>::name() + _(" -> ") +
            type_caster<retval_type>::name() +
            _(">"));
};

NAMESPACE_END(detail)
NAMESPACE_END(pybind11)
