/*
    pybind11/detail/numpy_ufunc.h: Simple glue for Python UFuncs

    Copyright (c) 2018 Eric Cousineau <eric.cousineau@tri.global>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "numpy.h"
#include "detail/inference.h"

NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
NAMESPACE_BEGIN(detail)

// Utilities

// Builtins registered using numpy/build/{...}/numpy/core/include/numpy/__umath_generated.c

template <typename... Args>
struct ufunc_ptr {
  PyUFuncGenericFunction func{};
  void* data{};
};

// Unary ufunc.
template <typename Arg0, typename Out, typename Func>
auto ufunc_to_ptr(Func func, type_pack<Arg0, Out>) {
    auto ufunc = [](
            char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
        Func& func = *(Func*)data;
        int step_0 = steps[0];
        int step_out = steps[1];
        int n = *dimensions;
        char *in_0 = args[0], *out = args[1];
        for (int k = 0; k < n; k++) {
            // TODO(eric.cousineau): Support pointers being changed.
            *(Out*)out = func(*(Arg0*)in_0);
            in_0 += step_0;
            out += step_out;
        }
    };
    // N.B. `new Func(...)` will never be destroyed.
    return ufunc_ptr<Arg0, Out>{ufunc, new Func(func)};
}

// Binary ufunc.
template <typename Arg0, typename Arg1, typename Out, typename Func = void>
auto ufunc_to_ptr(Func func, type_pack<Arg0, Arg1, Out>) {
    auto ufunc = [](char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
        Func& func = *(Func*)data;
        int step_0 = steps[0];
        int step_1 = steps[1];
        int step_out = steps[2];
        int n = *dimensions;
        char *in_0 = args[0], *in_1 = args[1], *out = args[2];
        for (int k = 0; k < n; k++) {
            // TODO(eric.cousineau): Support pointers being fed in.
            *(Out*)out = func(*(Arg0*)in_0, *(Arg1*)in_1);
            in_0 += step_0;
            in_1 += step_1;
            out += step_out;
        }
    };
    // N.B. `new Func(...)` will never be destroyed.
    return ufunc_ptr<Arg0, Arg1, Out>{ufunc, new Func(func)};
}

// Generic dispatch.
template <typename Func>
auto ufunc_to_ptr(Func func) {
    auto info = detail::function_inference::run(func);
    using Info = decltype(info);
    auto type_args = type_pack_apply<std::decay_t>(
        type_pack_concat(
            typename Info::Args{},
            type_pack<typename Info::Return>{}));
    return ufunc_to_ptr(func, type_args);
}

template <typename From, typename To, typename Func>
void ufunc_register_cast(
    Func&& func, bool allow_coercion, type_pack<From, To> = {}) {
  static auto cast_lambda = detail::function_inference::run(func).func;
  auto cast_func = +[](
        void* from_, void* to_, npy_intp n,
        void* fromarr, void* toarr) {
      const From* from = (From*)from_;
      To* to = (To*)to_;
      for (npy_intp i = 0; i < n; i++)
          to[i] = cast_lambda(from[i]);
  };
  auto& api = npy_api::get();
  auto from = npy_format_descriptor<From>::dtype();
  int to_num = npy_format_descriptor<To>::dtype().num();
  auto from_raw = (PyArray_Descr*)from.ptr();
  if (api.PyArray_RegisterCastFunc_(from_raw, to_num, cast_func) < 0)
      pybind11_fail("ufunc: Cannot register cast");
  if (allow_coercion) {
    if (api.PyArray_RegisterCanCast_(
            from_raw, to_num, npy_api::NPY_NOSCALAR_) < 0)
        pybind11_fail(
            "ufunc: Cannot register implicit / coercion cast capability");
  }
}

NAMESPACE_END(detail)

class ufunc : public object {
public:
    ufunc(object ptr) : object(ptr) {
        // TODO(eric.cousineau): Check type.
    }

    ufunc(detail::PyUFuncObject* ptr)
        : object(reinterpret_borrow<object>((PyObject*)ptr))
    {}

    ufunc(handle scope, const char* name) : scope_{scope}, name_{name} {}

    // Gets a NumPy UFunc by name.
    static ufunc get_builtin(const char* name) {
        module numpy = module::import("numpy");
        return ufunc(numpy.attr(name));
    }

    template <typename Type, typename Func>
    ufunc& def_loop(Func func) {
        do_register<Type>(detail::ufunc_to_ptr(func));
        return *this;
    }

    detail::PyUFuncObject* ptr() const {
        return (detail::PyUFuncObject*)self().ptr();
    }

private:
    object& self() { return *this; }
    const object& self() const { return *this; }

    // Registers a function pointer as a UFunc, mapping types to dtype nums.
    template <typename Type, typename ... Args>
    void do_register(detail::ufunc_ptr<Args...> user) {
        constexpr int N = sizeof...(Args);
        constexpr int nin = N - 1;
        constexpr int nout = 1;
        int dtype = dtype::of<Type>().num();
        int dtype_args[] = {dtype::of<Args>().num()...};
        // Determine if we need to make a new ufunc.
        using constants = detail::npy_api::constants;
        auto& api = detail::npy_api::get();
        if (!self()) {
            if (!name_)
                pybind11_fail("dtype: unspecified name");
            // TODO(eric.cousineau): Fix unfreed memory with `name`.
            auto leak = new std::string(name_);
            // The following dummy stuff is to allow monkey-patching existing ufuncs.
            // This is a bit sketchy, as calling the wrong thing may cause a segfault.
            // TODO(eric.cousineau): Figure out how to more elegantly specify preallocation...
            // Preallocate to allow replacement?
            constexpr int ntypes = 4;
            static char tinker_types[ntypes] = {
                constants::NPY_BOOL_,
                constants::NPY_INT_,
                constants::NPY_FLOAT_,
                constants::NPY_DOUBLE_,
            };
            auto dummy_funcs = new detail::PyUFuncGenericFunction[ntypes];
            auto dummy_data = new void*[ntypes];
            constexpr int ntotal = (nin + nout) * ntypes;
            auto dummy_types = new char[ntotal];
            for (int it = 0; it < ntypes; ++it) {
                for (int iarg = 0; iarg < nin + nout; ++iarg) {
                    int i = it * (nin + nout) + iarg;
                    dummy_types[i] = tinker_types[it];
                }
            }
            auto h = api.PyUFunc_FromFuncAndData_(
                dummy_funcs, dummy_data, dummy_types, ntypes,
                nin, nout, constants::PyUFunc_None_, &(*leak)[0], "", 0);
            self() = reinterpret_borrow<object>((PyObject*)h);
            scope_.attr(name_) = self();
        }
        if (N != ptr()->nargs)
            pybind11_fail("ufunc: Argument count mismatch");
        if (dtype >= constants::NPY_USERDEF_) {
            if (api.PyUFunc_RegisterLoopForType_(
                    ptr(), dtype, user.func, dtype_args, user.data) < 0)
                pybind11_fail("ufunc: Failed to register custom ufunc");
        } else {
            // Hack because NumPy API doesn't allow us convenience for builtin types :(
            if (api.PyUFunc_ReplaceLoopBySignature_(
                    ptr(), user.func, dtype_args, nullptr) < 0)
                pybind11_fail("ufunc: Failed ot register builtin ufunc");
            // Now that we've registered, ensure that we replace the data.
            bool found{};
            for (int i = 0; i < ptr()->ntypes; ++i) {
                if (ptr()->functions[i] == user.func) {
                    found = true;
                    ptr()->data[i] = user.data;
                    break;
                }
            }
            if (!found)
                pybind11_fail("Can't hack and slash");
        }
    }

    // These are only used if we have something new.
    const char* name_{};
    handle scope_{}; 
};

NAMESPACE_END(PYBIND11_NAMESPACE)
