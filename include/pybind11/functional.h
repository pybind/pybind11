/*
    pybind11/functional.h: std::function<> support

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#define PYBIND11_HAS_TYPE_CASTER_STD_FUNCTION_SPECIALIZATIONS

#include "pybind11.h"

#include <functional>
#include <iostream>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)
PYBIND11_NAMESPACE_BEGIN(type_caster_std_function_specializations)

// ensure GIL is held during functor destruction
struct func_handle {
    function f;
#if !(defined(_MSC_VER) && _MSC_VER == 1916 && defined(PYBIND11_CPP17))
    // This triggers a syntax error under very special conditions (very weird indeed).
    explicit
#endif
        func_handle(function &&f_) noexcept
        : f(std::move(f_)) {
    }
    func_handle(const func_handle &f_) { operator=(f_); }
    func_handle &operator=(const func_handle &f_) {
        gil_scoped_acquire acq;
        f = f_.f;
        return *this;
    }
    ~func_handle() {
        gil_scoped_acquire acq;
        function kill_f(std::move(f));
    }
};

// to emulate 'move initialization capture' in C++11
struct func_wrapper_base {
    func_handle hfunc;
    explicit func_wrapper_base(func_handle &&hf) noexcept : hfunc(hf) {}
};

template <typename Return, typename... Args>
struct func_wrapper : func_wrapper_base {
    using func_wrapper_base::func_wrapper_base;
    Return operator()(Args... args) const {
        gil_scoped_acquire acq;
        // casts the returned object as a rvalue to the return type
        return hfunc.f(std::forward<Args>(args)...).template cast<Return>();
    }
};

PYBIND11_NAMESPACE_END(type_caster_std_function_specializations)

template <typename Return, typename... Args>
struct type_caster<std::function<Return(Args...)>> {
    using type = std::function<Return(Args...)>;
    using retval_type = conditional_t<std::is_same<Return, void>::value, void_type, Return>;
    using function_type = Return (*)(Args...);

public:
    bool load(handle src, bool convert) {
        if (src.is_none()) {
            // Defer accepting None to other overloads (if we aren't in convert mode):
            if (!convert) {
                return false;
            }
            return true;
        }

        if (!isinstance<function>(src)) {
            return false;
        }

        auto func = reinterpret_borrow<function>(src);

        /*
           When passing a C++ function as an argument to another C++
           function via Python, every function call would normally involve
           a full C++ -> Python -> C++ roundtrip, which can be prohibitive.
           Here, we try to at least detect the case where the function is
           stateless (i.e. function pointer or lambda function without
           captured variables), in which case the roundtrip can be avoided.
         */
        if (auto cfunc = func.cpp_function()) {
            auto *cfunc_self = PyCFunction_GET_SELF(cfunc.ptr());
            if (cfunc_self == nullptr) {
                PyErr_Clear();
            } else if (isinstance<capsule>(cfunc_self)) {
                auto c = reinterpret_borrow<capsule>(cfunc_self);

                function_record *rec = nullptr;
                // Check that we can safely reinterpret the capsule into a function_record
                if (detail::is_function_record_capsule(c)) {
                    rec = c.get_pointer<function_record>();
                }
                while (rec != nullptr) {
                    const size_t self_offset = rec->is_method ? 1 : 0;
                    if (rec->nargs != sizeof...(Args) + self_offset) {
                        rec = rec->next;
                        // if the overload is not feasible in terms of number of arguments, we
                        // continue to the next one. If there is no next one, we return false.
                        if (rec == nullptr) {
                            return false;
                        }
                        continue;
                    }
                    if (rec->is_stateless
                        && same_type(typeid(function_type),
                                     *reinterpret_cast<const std::type_info *>(rec->data[1]))) {
                        struct capture {
                            function_type f;
                        };
                        value = ((capture *) &rec->data)->f;
                        return true;
                    }
                    rec = rec->next;
                }
            }
            // PYPY segfaults here when passing builtin function like sum.
            // Raising an fail exception here works to prevent the segfault, but only on gcc.
            // See PR #1413 for full details
        } else {
            // Check number of arguments of Python function
            auto get_argument_count = [](const handle &obj) -> size_t {
                // Faster then `import inspect` and `inspect.signature(obj).parameters`
                return obj.attr("co_argcount").cast<size_t>();
            };
            size_t argCount = 0;

            handle empty;
            object codeAttr = getattr(src, "__code__", empty);

            if (codeAttr) {
                argCount = get_argument_count(codeAttr);
            } else {
                object callAttr = getattr(src, "__call__", empty);

                if (callAttr) {
                    object codeAttr2 = getattr(callAttr, "__code__");
                    argCount = get_argument_count(codeAttr2) - 1; // removing the self argument
                } else {
                    // No __code__ or __call__ attribute, this is not a proper Python function
                    return false;
                }
            }
            // if we are a method, we have to correct the argument count since we are not counting
            // the self argument
            const size_t self_offset = static_cast<bool>(PyMethod_Check(src.ptr())) ? 1 : 0;

            argCount -= self_offset;
            if (argCount != sizeof...(Args)) {
                return false;
            }
        }

        value = type_caster_std_function_specializations::func_wrapper<Return, Args...>(
            type_caster_std_function_specializations::func_handle(std::move(func)));
        return true;
    }

    template <typename Func>
    static handle cast(Func &&f_, return_value_policy policy, handle /* parent */) {
        if (!f_) {
            return none().release();
        }

        auto result = f_.template target<function_type>();
        if (result) {
            return cpp_function(*result, policy).release();
        }
        return cpp_function(std::forward<Func>(f_), policy).release();
    }

    PYBIND11_TYPE_CASTER(type,
                         const_name("Callable[[")
                             + ::pybind11::detail::concat(make_caster<Args>::name...)
                             + const_name("], ") + make_caster<retval_type>::name
                             + const_name("]"));
};

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
