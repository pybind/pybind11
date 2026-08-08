/*
    pybind11/pybind11.h: Main header file of the C++11 python
    binding generator library

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once
#include "detail/class.h"
#include "detail/dynamic_raw_ptr_cast_if_possible.h"
#include "detail/exception_translation.h"
#include "detail/function_record_pyobject.h"
#include "detail/function_ref.h"
#include "detail/init.h"
#include "detail/native_enum_data.h"
#include "detail/using_smart_holder.h"
#include "attr.h"
#include "gil.h"
#include "gil_safe_call_once.h"
#include "options.h"
#include "trampoline_self_life_support.h"
#include "typing.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stack>
#include <string>
#include <utility>
#include <vector>

// See PR #5448. This warning suppression is needed for the PYBIND11_OVERRIDE macro family.
// NOTE that this is NOT embedded in a push/pop pair because that is very difficult to achieve.
#if defined(__clang_major__) && __clang_major__ < 14
PYBIND11_WARNING_DISABLE_CLANG("-Wgnu-zero-variadic-macro-arguments")
#endif

#if defined(__GNUG__) && !defined(__clang__)
#    include <cxxabi.h>
#endif

#if defined(__cpp_if_constexpr) && __cpp_if_constexpr >= 201606
#    define PYBIND11_MAYBE_CONSTEXPR constexpr
#else
#    define PYBIND11_MAYBE_CONSTEXPR
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

/* https://stackoverflow.com/questions/46798456/handling-gccs-noexcept-type-warning
   This warning is about ABI compatibility, not code health.
   It is only actually needed in a couple places, but apparently GCC 7 "generates this warning if
   and only if the first template instantiation ... involves noexcept" [stackoverflow], therefore
   it could get triggered from seemingly random places, depending on user code.
   No other GCC version generates this warning.
 */
#if defined(__GNUC__) && __GNUC__ == 7
PYBIND11_WARNING_DISABLE_GCC("-Wnoexcept-type")
#endif

PYBIND11_WARNING_DISABLE_MSVC(4127)

PYBIND11_NAMESPACE_BEGIN(detail)

std::string replace_newlines_and_squash(const char *text);

/* Generate a proper function signature */
std::string generate_function_signature(const char *type_caster_name_field,
                                        detail::function_record *func_rec,
                                        const std::type_info *const *types,
                                        size_t &type_index,
                                        size_t &arg_index);

template <typename T>
inline std::string generate_type_signature() {
    static constexpr auto caster_name_field = make_caster<T>::name;
    PYBIND11_DESCR_CONSTEXPR auto descr_types = decltype(caster_name_field)::types();
    // Create a default function_record to ensure the function signature has the proper
    // configuration e.g. no_convert.
    auto func_rec = function_record();
    size_t type_index = 0;
    size_t arg_index = 0;
    return generate_function_signature(
        caster_name_field.text, &func_rec, descr_types.data(), type_index, arg_index);
}

#if defined(_MSC_VER)
#    define PYBIND11_COMPAT_STRDUP _strdup
#else
#    define PYBIND11_COMPAT_STRDUP strdup
#endif

#define PYBIND11_READABLE_FUNCTION_SIGNATURE_EXPR                                                 \
    detail::const_name("(") + cast_in::arg_names + detail::const_name(") -> ") + cast_out::name

// We factor out readable function signatures to a specific template
// so that they don't get duplicated across different instantiations of
// cpp_function::initialize (which is templated on more types).
template <typename cast_in, typename cast_out>
class ReadableFunctionSignature {
public:
    using sig_type = decltype(PYBIND11_READABLE_FUNCTION_SIGNATURE_EXPR);

private:
    // We have to repeat PYBIND11_READABLE_FUNCTION_SIGNATURE_EXPR in decltype()
    // because C++11 doesn't allow functions to return `auto`. (We don't
    // know the type because it's some variant of detail::descr<N> with
    // unknown N.)
    static constexpr sig_type sig() { return PYBIND11_READABLE_FUNCTION_SIGNATURE_EXPR; }

public:
    static constexpr sig_type kSig = sig();
    // We can only stash the result of detail::descr::types() in a
    // constexpr variable if we aren't on MSVC (see
    // PYBIND11_DESCR_CONSTEXPR).
#if !defined(_MSC_VER)
    using types_type = decltype(sig_type::types());
    static constexpr types_type kTypes = sig_type::types();
#endif
};
#undef PYBIND11_READABLE_FUNCTION_SIGNATURE_EXPR

// Prior to C++17, we don't have inline variables, so we have to
// provide an out-of-line definition of the class member.
#if !defined(PYBIND11_CPP17)
template <typename cast_in, typename cast_out>
constexpr typename ReadableFunctionSignature<cast_in, cast_out>::sig_type
    ReadableFunctionSignature<cast_in, cast_out>::kSig;
#    if !defined(_MSC_VER)
template <typename cast_in, typename cast_out>
constexpr typename ReadableFunctionSignature<cast_in, cast_out>::types_type
    ReadableFunctionSignature<cast_in, cast_out>::kTypes;
#    endif
#endif

PYBIND11_NAMESPACE_END(detail)

/// Wraps an arbitrary C++ function/method/lambda function/.. into a callable Python object
class cpp_function : public function {
public:
    cpp_function() = default;
    // NOLINTNEXTLINE(google-explicit-constructor)
    cpp_function(std::nullptr_t) {}
    cpp_function(std::nullptr_t, const is_setter &) {}

    /// Construct a cpp_function from a vanilla function pointer
    template <typename Return, typename... Args, typename... Extra>
    // NOLINTNEXTLINE(google-explicit-constructor)
    cpp_function(Return (*f)(Args...), const Extra &...extra) {
        initialize(f, f, extra...);
    }

    /// Construct a cpp_function from a lambda function (possibly with internal state)
    template <typename Func,
              typename... Extra,
              typename = detail::enable_if_t<detail::is_lambda<Func>::value>>
    // NOLINTNEXTLINE(google-explicit-constructor)
    cpp_function(Func &&f, const Extra &...extra) {
        initialize(
            std::forward<Func>(f), (detail::function_signature_t<Func> *) nullptr, extra...);
    }

    /// Construct a cpp_function from a class method (non-const, no ref-qualifier)
    template <typename Return, typename Class, typename... Arg, typename... Extra>
    // NOLINTNEXTLINE(google-explicit-constructor)
    cpp_function(Return (Class::*f)(Arg...), const Extra &...extra) {
        initialize(
            [f](Class *c, Arg... args) -> Return { return (c->*f)(std::forward<Arg>(args)...); },
            (Return (*)(Class *, Arg...)) nullptr,
            extra...);
    }

    /// Construct a cpp_function from a class method (non-const, lvalue ref-qualifier)
    /// A copy of the overload for non-const functions without explicit ref-qualifier
    /// but with an added `&`.
    template <typename Return, typename Class, typename... Arg, typename... Extra>
    // NOLINTNEXTLINE(google-explicit-constructor)
    cpp_function(Return (Class::*f)(Arg...) &, const Extra &...extra) {
        initialize(
            [f](Class *c, Arg... args) -> Return { return (c->*f)(std::forward<Arg>(args)...); },
            (Return (*)(Class *, Arg...)) nullptr,
            extra...);
    }

    /// Construct a cpp_function from a class method (const, no ref-qualifier)
    template <typename Return, typename Class, typename... Arg, typename... Extra>
    // NOLINTNEXTLINE(google-explicit-constructor)
    cpp_function(Return (Class::*f)(Arg...) const, const Extra &...extra) {
        initialize([f](const Class *c,
                       Arg... args) -> Return { return (c->*f)(std::forward<Arg>(args)...); },
                   (Return (*)(const Class *, Arg...)) nullptr,
                   extra...);
    }

    /// Construct a cpp_function from a class method (const, lvalue ref-qualifier)
    /// A copy of the overload for const functions without explicit ref-qualifier
    /// but with an added `&`.
    template <typename Return, typename Class, typename... Arg, typename... Extra>
    // NOLINTNEXTLINE(google-explicit-constructor)
    cpp_function(Return (Class::*f)(Arg...) const &, const Extra &...extra) {
        initialize([f](const Class *c,
                       Arg... args) -> Return { return (c->*f)(std::forward<Arg>(args)...); },
                   (Return (*)(const Class *, Arg...)) nullptr,
                   extra...);
    }

    /// Construct a cpp_function from a class method (non-const, rvalue ref-qualifier)
    template <typename Return, typename Class, typename... Arg, typename... Extra>
    // NOLINTNEXTLINE(google-explicit-constructor)
    cpp_function(Return (Class::*f)(Arg...) &&, const Extra &...extra) {
        initialize(
            [f](Class *c, Arg... args) -> Return {
                return (std::move(*c).*f)(std::forward<Arg>(args)...);
            },
            (Return (*)(Class *, Arg...)) nullptr,
            extra...);
    }

    /// Construct a cpp_function from a class method (const, rvalue ref-qualifier)
    template <typename Return, typename Class, typename... Arg, typename... Extra>
    // NOLINTNEXTLINE(google-explicit-constructor)
    cpp_function(Return (Class::*f)(Arg...) const &&, const Extra &...extra) {
        initialize(
            [f](const Class *c, Arg... args) -> Return {
                return (std::move(*c).*f)(std::forward<Arg>(args)...);
            },
            (Return (*)(const Class *, Arg...)) nullptr,
            extra...);
    }

#ifdef __cpp_noexcept_function_type
    /// Construct a cpp_function from a class method (non-const, no ref-qualifier, noexcept)
    template <typename Return, typename Class, typename... Arg, typename... Extra>
    // NOLINTNEXTLINE(google-explicit-constructor)
    cpp_function(Return (Class::*f)(Arg...) noexcept, const Extra &...extra) {
        initialize(
            [f](Class *c, Arg... args) -> Return { return (c->*f)(std::forward<Arg>(args)...); },
            (Return (*)(Class *, Arg...)) nullptr,
            extra...);
    }

    /// Construct a cpp_function from a class method (non-const, lvalue ref-qualifier, noexcept)
    template <typename Return, typename Class, typename... Arg, typename... Extra>
    // NOLINTNEXTLINE(google-explicit-constructor)
    cpp_function(Return (Class::*f)(Arg...) & noexcept, const Extra &...extra) {
        initialize(
            [f](Class *c, Arg... args) -> Return { return (c->*f)(std::forward<Arg>(args)...); },
            (Return (*)(Class *, Arg...)) nullptr,
            extra...);
    }

    /// Construct a cpp_function from a class method (const, no ref-qualifier, noexcept)
    template <typename Return, typename Class, typename... Arg, typename... Extra>
    // NOLINTNEXTLINE(google-explicit-constructor)
    cpp_function(Return (Class::*f)(Arg...) const noexcept, const Extra &...extra) {
        initialize([f](const Class *c,
                       Arg... args) -> Return { return (c->*f)(std::forward<Arg>(args)...); },
                   (Return (*)(const Class *, Arg...)) nullptr,
                   extra...);
    }

    /// Construct a cpp_function from a class method (const, lvalue ref-qualifier, noexcept)
    template <typename Return, typename Class, typename... Arg, typename... Extra>
    // NOLINTNEXTLINE(google-explicit-constructor)
    cpp_function(Return (Class::*f)(Arg...) const & noexcept, const Extra &...extra) {
        initialize([f](const Class *c,
                       Arg... args) -> Return { return (c->*f)(std::forward<Arg>(args)...); },
                   (Return (*)(const Class *, Arg...)) nullptr,
                   extra...);
    }

    /// Construct a cpp_function from a class method (non-const, rvalue ref-qualifier, noexcept)
    template <typename Return, typename Class, typename... Arg, typename... Extra>
    // NOLINTNEXTLINE(google-explicit-constructor)
    cpp_function(Return (Class::*f)(Arg...) && noexcept, const Extra &...extra) {
        initialize(
            [f](Class *c, Arg... args) -> Return {
                return (std::move(*c).*f)(std::forward<Arg>(args)...);
            },
            (Return (*)(Class *, Arg...)) nullptr,
            extra...);
    }

    /// Construct a cpp_function from a class method (const, rvalue ref-qualifier, noexcept)
    template <typename Return, typename Class, typename... Arg, typename... Extra>
    // NOLINTNEXTLINE(google-explicit-constructor)
    cpp_function(Return (Class::*f)(Arg...) const && noexcept, const Extra &...extra) {
        initialize(
            [f](const Class *c, Arg... args) -> Return {
                return (std::move(*c).*f)(std::forward<Arg>(args)...);
            },
            (Return (*)(const Class *, Arg...)) nullptr,
            extra...);
    }
#endif

    /// Return the function name
    object name() const { return attr("__name__"); }

protected:
    struct InitializingFunctionRecordDeleter {
        // `destruct(function_record, false)`: `initialize_generic` copies strings and
        // takes care of cleaning up in case of exceptions. So pass `false` to `free_strings`.
        void operator()(detail::function_record *rec) { destruct(rec, false); }
    };
    using unique_function_record
        = std::unique_ptr<detail::function_record, InitializingFunctionRecordDeleter>;

    /// Space optimization: don't inline this frequently instantiated fragment
    unique_function_record make_function_record();

private:
    // This is outlined from the dispatch lambda in initialize to save
    // on code size. Crucially, we use function_ref to type-erase the
    // actual function lambda so that we can get code reuse for
    // functions with the same Return, Args, and Guard.
    template <typename Return, typename Guard, typename ArgsConverter, typename... Args>
    static handle call_impl(detail::function_call &call, detail::function_ref<Return(Args...)> f) {
        using namespace detail;
        // Static assertion: function_ref must be trivially copyable to ensure safe pass-by-value.
        // Lifetime safety: The function_ref is created from cap->f which lives in the capture
        // object stored in the function record, and is only used synchronously within this
        // function call. It is never stored beyond the scope of call_impl.
        static_assert(std::is_trivially_copyable<detail::function_ref<Return(Args...)>>::value,
                      "function_ref must be trivially copyable for safe pass-by-value usage");
        using cast_out
            = make_caster<conditional_t<std::is_void<Return>::value, void_type, Return>>;

        ArgsConverter args_converter;
        if (!args_converter.load_args(call)) {
            return PYBIND11_TRY_NEXT_OVERLOAD;
        }

        /* Override policy for rvalues -- usually to enforce rvp::move on an rvalue */
        return_value_policy policy
            = return_value_policy_override<Return>::policy(call.func.policy);

        /* Perform the function call */
        handle result;
        if (call.func.is_setter) {
            (void) std::move(args_converter).template call<Return, Guard>(f);
            result = none().release();
        } else {
            result = cast_out::cast(
                std::move(args_converter).template call<Return, Guard>(f), policy, call.parent);
        }

        return result;
    }

protected:
    /// Special internal constructor for functors, lambda functions, etc.
    template <typename Func, typename Return, typename... Args, typename... Extra>
    void initialize(Func &&f, Return (*)(Args...), const Extra &...extra) {
        using namespace detail;
        struct capture {
            remove_reference_t<Func> f;

            static capture *from_data(void **data) {
                return PYBIND11_STD_LAUNDER(reinterpret_cast<capture *>(data));
            }
        };

        /* Store the function including any extra state it might have (e.g. a lambda capture
         * object) */
        // The unique_ptr makes sure nothing is leaked in case of an exception.
        auto unique_rec = make_function_record();
        auto *rec = unique_rec.get();

        /* Store the capture object directly in the function record if there is enough space */
        if (sizeof(capture) <= sizeof(rec->data)) {
            /* Without these pragmas, GCC warns that there might not be
               enough space to use the placement new operator. However, the
               'if' statement above ensures that this is the case. */
            PYBIND11_WARNING_PUSH

#if defined(__GNUG__) && __GNUC__ >= 6
            PYBIND11_WARNING_DISABLE_GCC("-Wplacement-new")
#endif

            new (capture::from_data(rec->data)) capture{std::forward<Func>(f)};

#if !PYBIND11_HAS_STD_LAUNDER
            PYBIND11_WARNING_DISABLE_GCC("-Wstrict-aliasing")
#endif

            // UB without std::launder, but without breaking ABI and/or
            // a significant refactoring it's "impossible" to solve.
            if (!std::is_trivially_destructible<capture>::value) {
                rec->free_data = [](function_record *r) {
                    auto data = capture::from_data(r->data);
                    (void) data; // suppress "unused variable" warnings
                    data->~capture();
                };
            }
            PYBIND11_WARNING_POP
        } else {
            rec->data[0] = new capture{std::forward<Func>(f)};
            rec->free_data = [](function_record *r) { delete ((capture *) r->data[0]); };
        }

        /* Type casters for the function arguments and return value */
        using cast_in = argument_loader<Args...>;
        using cast_out
            = make_caster<conditional_t<std::is_void<Return>::value, void_type, Return>>;

        static_assert(
            expected_num_args<Extra...>(
                sizeof...(Args), cast_in::args_pos >= 0, cast_in::has_kwargs),
            "The number of argument annotations does not match the number of function arguments");

        /* Dispatch code which converts function arguments and performs the actual function call */
        rec->impl = [](function_call &call) -> handle {
            /* Invoke call policy pre-call hook */
            process_attributes<Extra...>::precall(call);

            /* Get a pointer to the capture object */
            const auto *data = (sizeof(capture) <= sizeof(call.func.data) ? &call.func.data
                                                                          : call.func.data[0]);
            auto *cap = const_cast<capture *>(reinterpret_cast<const capture *>(data));

            auto result = call_impl<Return,
                                    /* Function scope guard -- defaults to the compile-to-nothing
                                       `void_type` */
                                    extract_guard_t<Extra...>,
                                    cast_in>(call, detail::function_ref<Return(Args...)>(cap->f));

            /* Invoke call policy post-call hook */
            process_attributes<Extra...>::postcall(call, result);

            return result;
        };

        rec->nargs_pos = cast_in::args_pos >= 0
                             ? static_cast<std::uint16_t>(cast_in::args_pos)
                             : sizeof...(Args) - cast_in::has_kwargs; // Will get reduced more if
                                                                      // we have a kw_only
        rec->has_args = cast_in::args_pos >= 0;
        rec->has_kwargs = cast_in::has_kwargs;

        /* Process any user-provided function attributes */
        process_attributes<Extra...>::init(extra..., rec);

        {
            constexpr bool has_kw_only_args = any_of<std::is_same<kw_only, Extra>...>::value,
                           has_pos_only_args = any_of<std::is_same<pos_only, Extra>...>::value,
                           has_arg_annotations = any_of<is_keyword<Extra>...>::value;
            constexpr bool has_is_method = any_of<std::is_same<is_method, Extra>...>::value;
            // The implicit `self` argument is not present and not counted in method definitions.
            constexpr bool has_args = cast_in::args_pos >= 0;
            constexpr bool is_method_with_self_arg_only = has_is_method && !has_args;
            static_assert(has_arg_annotations || !has_kw_only_args,
                          "py::kw_only requires the use of argument annotations");
            static_assert(((/* Need `py::arg("arg_name")` annotation in function/method. */
                            has_arg_annotations)
                           || (/* Allow methods with no arguments `def method(self, /): ...`.
                                * A method has at least one argument `self`. There can be no
                                * `py::arg` annotation. E.g. `class.def("method", py::pos_only())`.
                                */
                               is_method_with_self_arg_only))
                              || !has_pos_only_args,
                          "py::pos_only requires the use of argument annotations (for docstrings "
                          "and aligning the annotations to the argument)");

            static_assert(constexpr_sum(is_kw_only<Extra>::value...) <= 1,
                          "py::kw_only may be specified only once");
            static_assert(constexpr_sum(is_pos_only<Extra>::value...) <= 1,
                          "py::pos_only may be specified only once");
            constexpr auto kw_only_pos = constexpr_first<is_kw_only, Extra...>();
            constexpr auto pos_only_pos = constexpr_first<is_pos_only, Extra...>();
            static_assert(!(has_kw_only_args && has_pos_only_args) || pos_only_pos < kw_only_pos,
                          "py::pos_only must come before py::kw_only");
        }

        /* Generate a readable signature describing the function's arguments and return
           value types */
        static constexpr const auto &signature
            = detail::ReadableFunctionSignature<cast_in, cast_out>::kSig;
#if !defined(_MSC_VER)
        static constexpr const auto &types
            = detail::ReadableFunctionSignature<cast_in, cast_out>::kTypes;
#else
        PYBIND11_DESCR_CONSTEXPR auto types = std::decay<decltype(signature)>::type::types();
#endif

        /* Register the function with Python from generic (non-templated) code */
        // Pass on the ownership over the `unique_rec` to `initialize_generic`. `rec` stays valid.
        initialize_generic(std::move(unique_rec), signature.text, types.data(), sizeof...(Args));

        /* Stash some additional information used by an important optimization in 'functional.h' */
        using FunctionType = Return (*)(Args...);
        constexpr bool is_function_ptr
            = std::is_convertible<Func, FunctionType>::value && sizeof(capture) == sizeof(void *);
        PYBIND11_ENSURE_PRECONDITION_FOR_FUNCTIONAL_H_PERFORMANCE_OPTIMIZATIONS(
            !is_function_ptr || std::is_standard_layout<capture>::value);
        if (is_function_ptr) {
            rec->is_stateless = true;
            rec->data[1]
                = const_cast<void *>(reinterpret_cast<const void *>(&typeid(FunctionType)));
        }
    }

    // Utility class that keeps track of all duplicated strings, and cleans them up in its
    // destructor, unless they are released. Basically a RAII-solution to deal with exceptions
    // along the way.
    class strdup_guard {
    public:
        strdup_guard() = default;
        strdup_guard(const strdup_guard &) = delete;
        strdup_guard &operator=(const strdup_guard &) = delete;

        ~strdup_guard() {
            for (auto *s : strings) {
                std::free(s);
            }
        }
        char *operator()(const char *s) {
            auto *t = PYBIND11_COMPAT_STRDUP(s);
            strings.push_back(t);
            return t;
        }
        void release() { strings.clear(); }

    private:
        std::vector<char *> strings;
    };

    /// Register a function call with Python (generic non-templated code goes here)
    void initialize_generic(unique_function_record &&unique_rec,
                            const char *text,
                            const std::type_info *const *types,
                            size_t args);

    friend void detail::function_record_PyTypeObject_methods::tp_dealloc_impl(PyObject *);

    /// When a cpp_function is GCed, release any memory allocated by pybind11
    static void destruct(detail::function_record *rec, bool free_strings = true);

    /// Main dispatch logic for calls to functions bound using pybind11
    static PyObject *
    dispatcher(PyObject *self, PyObject *const *args_in_arr, size_t nargsf, PyObject *kwnames_in);

    static ssize_t keyword_index(PyObject *haystack, char const *needle) {
        /* kwargs is usually very small (<= 5 entries).  The arg strings are typically interned.
         * CPython itself implements the search this way, first comparing all pointers ... which is
         * cheap and will work if the strings are interned.  If it fails, then it falls back to a
         * second lexicographic check. This is wildly expensive for huge argument lists, but those
         * are incredibly rare so we optimize for the vastly common case of just a couple of args.
         */
        auto n = PyTuple_GET_SIZE(haystack);
        auto s = reinterpret_steal<pybind11::str>(PyUnicode_InternFromString(needle));
        for (ssize_t i = 0; i < n; ++i) {
            if (PyTuple_GET_ITEM(haystack, i) == s.ptr()) {
                return i;
            }
        }
        for (ssize_t i = 0; i < n; ++i) {
            if (PyUnicode_Compare(PyTuple_GET_ITEM(haystack, i), s.ptr()) == 0) {
                return i;
            }
        }
        return -1;
    }
};

PYBIND11_NAMESPACE_BEGIN(detail)

template <>
struct handle_type_name<cpp_function> {
    static constexpr auto name = const_name("collections.abc.Callable");
};

PYBIND11_NAMESPACE_END(detail)

// Use to activate Py_MOD_GIL_NOT_USED.
class mod_gil_not_used {
public:
    mod_gil_not_used() : flag_(true) {}
    PYBIND11_DEPRECATED("use py::mod_gil_not_used() or py::mod_gil_used() instead")
    explicit mod_gil_not_used(bool flag) : flag_(flag) {}
    bool flag() const { return flag_; }

    friend mod_gil_not_used mod_gil_used();

private:
    bool flag_;
};

// Use to activate Py_MOD_GIL_USED, the current default.
inline mod_gil_not_used mod_gil_used() {
    mod_gil_not_used tag;
    tag.flag_ = false;
    return tag;
}

class multiple_interpreters {
public:
    enum class level {
        not_supported,      /// Use to activate Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED
        shared_gil,         /// Use to activate Py_MOD_MULTIPLE_INTERPRETERS_SUPPORTED
        per_interpreter_gil /// Use to activate Py_MOD_PER_INTERPRETER_GIL_SUPPORTED
    };

    static multiple_interpreters not_supported() {
        return multiple_interpreters(level::not_supported);
    }
    static multiple_interpreters shared_gil() { return multiple_interpreters(level::shared_gil); }
    static multiple_interpreters per_interpreter_gil() {
        return multiple_interpreters(level::per_interpreter_gil);
    }

    explicit constexpr multiple_interpreters(level l) : level_(l) {}
    level value() const { return level_; }

private:
    level level_;
};

PYBIND11_NAMESPACE_BEGIN(detail)

inline bool gil_not_used_option() { return false; }
template <typename F, typename... O>
bool gil_not_used_option(F &&, O &&...o);
template <typename... O>
inline bool gil_not_used_option(mod_gil_not_used f, O &&...o) {
    return f.flag() || gil_not_used_option(o...);
}
template <typename F, typename... O>
inline bool gil_not_used_option(F &&, O &&...o) {
    return gil_not_used_option(o...);
}

#ifdef Py_mod_multiple_interpreters
inline void *multi_interp_slot() { return Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED; }
template <typename... O>
inline void *multi_interp_slot(multiple_interpreters mi, O &&...o) {
    switch (mi.value()) {
        case multiple_interpreters::level::per_interpreter_gil:
            return Py_MOD_PER_INTERPRETER_GIL_SUPPORTED;
        case multiple_interpreters::level::shared_gil:
            return Py_MOD_MULTIPLE_INTERPRETERS_SUPPORTED;
        case multiple_interpreters::level::not_supported:
            return Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED;
    }
    // silence warnings with this unreachable line:
    return multi_interp_slot(o...);
}
template <typename F, typename... O>
inline void *multi_interp_slot(F &&, O &&...o) {
    return multi_interp_slot(o...);
}
#endif

/*
Return a borrowed reference to the named module if it has been successfully initialized within this
interpreter before. nullptr if it has not been successfully initialized.
*/
PyObject *get_cached_module(pybind11::str const &nameobj);

/*
Add successfully initialized a module object to the internal cache.

The module must have a __spec__ attribute with a name attribute.
*/
void cache_completed_module(pybind11::object const &mod);

/*
A Py_mod_create slot function which will return the previously created module from the cache if one
exists, and otherwise will create a new module object.
*/
PyObject *cached_create_module(PyObject *spec, PyModuleDef *);

/// Must be a POD type, and must hold enough entries for all of the possible slots PLUS ONE for
/// the sentinel (0) end slot.
using slots_array = std::array<PyModuleDef_Slot, 5>;

/// Initialize an array of slots based on the supplied exec slot and options.
template <typename... Options>
inline slots_array init_slots(int (*exec_fn)(PyObject *), Options &&...options) noexcept {
    /* NOTE: slots_array MUST be large enough to hold all possible options.  If you add an option
    here, you MUST also increase the size of slots_array in the type alias above! */
    slots_array mod_def_slots;
    size_t next_slot = 0;

    mod_def_slots[next_slot++] = {Py_mod_create, reinterpret_cast<void *>(&cached_create_module)};

    if (exec_fn != nullptr) {
        mod_def_slots[next_slot++] = {Py_mod_exec, reinterpret_cast<void *>(exec_fn)};
    }

#ifdef Py_mod_multiple_interpreters
    mod_def_slots[next_slot++] = {Py_mod_multiple_interpreters, multi_interp_slot(options...)};
#endif

    if (gil_not_used_option(options...)) {
#if defined(Py_mod_gil) && defined(Py_GIL_DISABLED)
        mod_def_slots[next_slot++] = {Py_mod_gil, Py_MOD_GIL_NOT_USED};
#endif
    }

    // slots must have a zero end sentinel
    mod_def_slots[next_slot++] = {0, nullptr};

    return mod_def_slots;
}

PYBIND11_NAMESPACE_END(detail)

/// Wrapper for Python extension modules
class module_ : public object {
public:
    PYBIND11_OBJECT_DEFAULT(module_, object, PyModule_Check)

    /// Create a new top-level Python module with the given name and docstring
    PYBIND11_DEPRECATED("Use PYBIND11_MODULE or module_::create_extension_module instead")
    explicit module_(const char *name, const char *doc = nullptr) {
        *this = create_extension_module(name, doc, new PyModuleDef());
    }

    /** \rst
        Create Python binding for a new function within the module scope. ``Func``
        can be a plain C++ function, a function pointer, or a lambda function. For
        details on the ``Extra&& ... extra`` argument, see section :ref:`extras`.
    \endrst */
    template <typename Func, typename... Extra>
    module_ &def(const char *name_, Func &&f, const Extra &...extra) {
        cpp_function func(std::forward<Func>(f),
                          name(name_),
                          scope(*this),
                          sibling(getattr(*this, name_, none())),
                          extra...);
        // NB: allow overwriting here because cpp_function sets up a chain with the intention of
        // overwriting (and has already checked internally that it isn't overwriting
        // non-functions).
        add_object(name_, func, true /* overwrite */);
        return *this;
    }

    /** \rst
        Create and return a new Python submodule with the given name and docstring.
        This also works recursively, i.e.

        .. code-block:: cpp

            py::module_ m("example", "pybind11 example plugin");
            py::module_ m2 = m.def_submodule("sub", "A submodule of 'example'");
            py::module_ m3 = m2.def_submodule("subsub", "A submodule of 'example.sub'");
    \endrst */
    module_ def_submodule(const char *name, const char *doc = nullptr) {
        const char *this_name = PyModule_GetName(m_ptr);
        if (this_name == nullptr) {
            throw error_already_set();
        }
        std::string full_name = std::string(this_name) + '.' + name;
        handle submodule = PyImport_AddModule(full_name.c_str());
        if (!submodule) {
            throw error_already_set();
        }
        auto result = reinterpret_borrow<module_>(submodule);
        if (doc && options::show_user_defined_docstrings()) {
            result.attr("__doc__") = pybind11::str(doc);
        }

#if defined(GRAALVM_PYTHON) && (!defined(GRAALPY_VERSION_NUM) || GRAALPY_VERSION_NUM < 0x190000)
        // GraalPy doesn't support PyModule_GetFilenameObject,
        // so getting by attribute (see PR #5584)
        handle this_module = m_ptr;
        if (object this_file = getattr(this_module, "__file__", none())) {
            result.attr("__file__") = this_file;
        }
#else
        handle this_file = PyModule_GetFilenameObject(m_ptr);
        if (this_file) {
            result.attr("__file__") = this_file;
        } else if (PyErr_ExceptionMatches(PyExc_SystemError) != 0) {
            PyErr_Clear();
        } else {
            throw error_already_set();
        }
#endif
        attr(name) = result;
        return result;
    }

    /// Import and return a module or throws `error_already_set`.
    static module_ import(const char *name) {
        PyObject *obj = PyImport_ImportModule(name);
        if (!obj) {
            throw error_already_set();
        }
        return reinterpret_steal<module_>(obj);
    }

    /// Reload the module or throws `error_already_set`.
    void reload() {
        PyObject *obj = PyImport_ReloadModule(ptr());
        if (!obj) {
            throw error_already_set();
        }
        *this = reinterpret_steal<module_>(obj);
    }

    /** \rst
        Adds an object to the module using the given name.  Throws if an object with the given name
        already exists.

        ``overwrite`` should almost always be false: attempting to overwrite objects that pybind11
        has established will, in most cases, break things.
    \endrst */
    PYBIND11_NOINLINE void add_object(const char *name, handle obj, bool overwrite = false) {
        if (!overwrite && hasattr(*this, name)) {
            pybind11_fail(
                "Error during initialization: multiple incompatible definitions with name \""
                + std::string(name) + "\"");
        }

        PyModule_AddObject(ptr(), name, obj.inc_ref().ptr() /* steals a reference */);
    }

    // DEPRECATED (since PR #5688): Use PyModuleDef directly instead.
    using module_def = PyModuleDef;

    /** \rst
        Create a new top-level module that can be used as the main module of a C extension.

        ``def`` should point to a statically allocated PyModuleDef.
    \endrst */
    static module_ create_extension_module(const char *name,
                                           const char *doc,
                                           PyModuleDef *def,
                                           mod_gil_not_used gil_not_used = mod_gil_used()) {
        // Placement new (not an allocation).
        new (def) PyModuleDef{/* m_base */ PyModuleDef_HEAD_INIT,
                              /* m_name */ name,
                              /* m_doc */ options::show_user_defined_docstrings() ? doc : nullptr,
                              /* m_size */ -1,
                              /* m_methods */ nullptr,
                              /* m_slots */ nullptr,
                              /* m_traverse */ nullptr,
                              /* m_clear */ nullptr,
                              /* m_free */ nullptr};
        auto *m = PyModule_Create(def);
        if (m == nullptr) {
            if (PyErr_Occurred()) {
                throw error_already_set();
            }
            pybind11_fail("Internal error in module_::create_extension_module()");
        }
        if (gil_not_used.flag()) {
#ifdef Py_GIL_DISABLED
            PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif
        }
        // TODO: Should be reinterpret_steal for Python 3, but Python also steals it again when
        //       returned from PyInit_...
        //       For Python 2, reinterpret_borrow was correct.
        return reinterpret_borrow<module_>(m);
    }
};

PYBIND11_NAMESPACE_BEGIN(detail)

template <>
struct handle_type_name<module_> {
    static constexpr auto name = const_name("types.ModuleType");
};

PYBIND11_NAMESPACE_END(detail)

// When inside a namespace (or anywhere as long as it's not the first item on a line),
// C++20 allows "module" to be used. This is provided for backward compatibility, and for
// simplicity, if someone wants to use py::module for example, that is perfectly safe.
using module = module_;

/// \ingroup python_builtins
/// Return a dictionary representing the global variables in the current execution frame,
/// or ``__main__.__dict__`` if there is no frame (usually when the interpreter is embedded).
dict globals();

PYBIND11_NAMESPACE_BEGIN(detail)
/// Generic support for creating new Python heap types
class generic_type : public object {
public:
    PYBIND11_OBJECT_DEFAULT(generic_type, object, PyType_Check)
protected:
    void initialize(const type_record &rec);

    /// Helper function which tags all parents of a type using mult. inheritance
    void mark_parents_nonsimple(PyTypeObject *value);

    void install_buffer_funcs(buffer_info *(*get_buffer)(PyObject *, void *),
                              void *get_buffer_data) {
        auto *type = reinterpret_cast<PyHeapTypeObject *>(m_ptr);
        auto *tinfo = detail::get_type_info(&type->ht_type);

        if (!type->ht_type.tp_as_buffer) {
            pybind11_fail("To be able to register buffer protocol support for the type '"
                          + get_fully_qualified_tp_name(tinfo->type)
                          + "' the associated class<>(..) invocation must "
                            "include the pybind11::buffer_protocol() annotation!");
        }

        tinfo->get_buffer = get_buffer;
        tinfo->get_buffer_data = get_buffer_data;
    }

    // rec_func must be set for either fget or fset.
    void def_property_static_impl(const char *name,
                                  handle fget,
                                  handle fset,
                                  detail::function_record *rec_func) {
        const auto is_static = (rec_func != nullptr) && !(rec_func->is_method && rec_func->scope);
        const auto has_doc = (rec_func != nullptr) && (rec_func->doc != nullptr)
                             && pybind11::options::show_user_defined_docstrings();
        auto property = handle(reinterpret_cast<PyObject *>(
            is_static ? get_internals().static_property_type : &PyProperty_Type));
        attr(name) = property(fget.ptr() ? fget : none(),
                              fset.ptr() ? fset : none(),
                              /*deleter*/ none(),
                              pybind11::str(has_doc ? rec_func->doc : ""));
    }
};

/// Set the pointer to operator new if it exists. The cast is needed because it can be overloaded.
template <typename T,
          typename = void_t<decltype(static_cast<void *(*) (size_t)>(T::operator new))>>
void set_operator_new(type_record *r) {
    r->operator_new = &T::operator new;
}

template <typename>
void set_operator_new(...) {}

template <typename T, typename SFINAE = void>
struct has_operator_delete : std::false_type {};
template <typename T>
struct has_operator_delete<T, void_t<decltype(static_cast<void (*)(void *)>(T::operator delete))>>
    : std::true_type {};
template <typename T, typename SFINAE = void>
struct has_operator_delete_size : std::false_type {};
template <typename T>
struct has_operator_delete_size<
    T,
    void_t<decltype(static_cast<void (*)(void *, size_t)>(T::operator delete))>> : std::true_type {
};
/// Call class-specific delete if it exists or global otherwise. Can also be an overload set.
template <typename T, enable_if_t<has_operator_delete<T>::value, int> = 0>
void call_operator_delete(T *p, size_t, size_t) {
    T::operator delete(p);
}
template <typename T,
          enable_if_t<!has_operator_delete<T>::value && has_operator_delete_size<T>::value, int>
          = 0>
void call_operator_delete(T *p, size_t s, size_t) {
    T::operator delete(p, s);
}

void call_operator_delete(void *p, size_t s, size_t a);

void add_class_method(object &cls, const char *name_, const cpp_function &cf);

/// Type trait to rebind a member function pointer's class to `Derived`, preserving all
/// cv/ref/noexcept qualifiers. The primary template has no `type` member, providing SFINAE
/// failure for unsupported member function pointer types. `source_class` holds the original
/// class for use in `is_accessible_base_of` checks.
template <typename Derived, typename T>
struct rebind_member_ptr {};

// Define one specialization per supported qualifier combination via a local macro.
// The qualifiers argument appears in type position, not expression position, so
// parenthesizing it would produce invalid C++.
// The no-qualifier specialization is written out explicitly to avoid invoking the macro with an
// empty argument, which triggers MSVC warning C4003.
template <typename Derived, typename Return, typename Class, typename... Args>
struct rebind_member_ptr<Derived, Return (Class::*)(Args...)> {
    using type = Return (Derived::*)(Args...);
    using source_class = Class;
};
// NOLINTBEGIN(bugprone-macro-parentheses)
#define PYBIND11_REBIND_MEMBER_PTR(qualifiers)                                                    \
    template <typename Derived, typename Return, typename Class, typename... Args>                \
    struct rebind_member_ptr<Derived, Return (Class::*)(Args...) qualifiers> {                    \
        using type = Return (Derived::*)(Args...) qualifiers;                                     \
        using source_class = Class;                                                               \
    }
PYBIND11_REBIND_MEMBER_PTR(const);
PYBIND11_REBIND_MEMBER_PTR(&);
PYBIND11_REBIND_MEMBER_PTR(const &);
PYBIND11_REBIND_MEMBER_PTR(&&);
PYBIND11_REBIND_MEMBER_PTR(const &&);
#ifdef __cpp_noexcept_function_type
PYBIND11_REBIND_MEMBER_PTR(noexcept);
PYBIND11_REBIND_MEMBER_PTR(const noexcept);
PYBIND11_REBIND_MEMBER_PTR(& noexcept);
PYBIND11_REBIND_MEMBER_PTR(const & noexcept);
PYBIND11_REBIND_MEMBER_PTR(&& noexcept);
PYBIND11_REBIND_MEMBER_PTR(const && noexcept);
#endif
#undef PYBIND11_REBIND_MEMBER_PTR
// NOLINTEND(bugprone-macro-parentheses)

/// Shared implementation body for all method_adaptor member-function-pointer overloads.
/// Asserts Base is accessible from Derived, then casts the member pointer.
template <typename Derived,
          typename T,
          typename Traits = rebind_member_ptr<Derived, T>,
          typename Adapted = typename Traits::type>
constexpr PYBIND11_ALWAYS_INLINE Adapted adapt_member_ptr(T pmf) {
    static_assert(
        detail::is_accessible_base_of<typename Traits::source_class, Derived>::value,
        "Cannot bind an inaccessible base class method; use a lambda definition instead");
    return pmf;
}

PYBIND11_NAMESPACE_END(detail)

/// Given a pointer to a member function, cast it to its `Derived` version.
/// For all other callables (lambdas, function pointers, etc.), forward unchanged.
///
/// Two overloads cover all cases without explicit per-qualifier instantiations:
///
///  (1) Generic fallback — disabled for member function pointers so that (2) wins
///      without any partial-ordering ambiguity.
///  (2) MFP overload — SFINAE on rebind_member_ptr::type, which exists for every
///      supported qualifier combination (const, &, &&, noexcept, ...).  A single
///      template therefore covers all combinations that rebind_member_ptr handles.
template <
    typename /*Derived*/,
    typename F,
    detail::enable_if_t<!std::is_member_function_pointer<detail::remove_reference_t<F>>::value,
                        int> = 0>
constexpr auto method_adaptor(F &&f) -> decltype(std::forward<F>(f)) {
    return std::forward<F>(f);
}

template <typename Derived,
          typename T,
          typename Adapted = typename detail::rebind_member_ptr<Derived, T>::type>
constexpr Adapted method_adaptor(T pmf) {
    // Expected to be redundant (SFINAE on rebind_member_ptr) but cheap and makes the intent
    // explicit.
    static_assert(std::is_member_function_pointer<T>::value,
                  "method_adaptor: T must be a member function pointer");
    return detail::adapt_member_ptr<Derived>(pmf);
}

PYBIND11_NAMESPACE_BEGIN(detail)

// Helper for the property_cpp_function static member functions below.
// The only purpose of these functions is to support .def_readonly & .def_readwrite.
// In this context, the PM template parameter is certain to be a Pointer to a Member.
// The main purpose of must_be_member_function_pointer is to make this obvious, and to guard
// against accidents. As a side-effect, it also explains why the syntactical overhead for
// perfect forwarding is not needed.
template <typename PM>
using must_be_member_function_pointer = enable_if_t<std::is_member_pointer<PM>::value, int>;

// Note that property_cpp_function is intentionally in the main pybind11 namespace,
// because user-defined specializations could be useful.

// Classic (non-smart_holder) implementations for .def_readonly and .def_readwrite
// getter and setter functions.
// WARNING: This classic implementation can lead to dangling pointers for raw pointer members.
// See test_ptr() in tests/test_class_sh_property.py
// However, this implementation works as-is (and safely) for smart_holder std::shared_ptr members.
template <typename T, typename D>
struct property_cpp_function_classic {
    template <typename PM, must_be_member_function_pointer<PM> = 0>
    static cpp_function readonly(PM pm, const handle &hdl) {
        return cpp_function([pm](const T &c) -> const D & { return c.*pm; }, is_method(hdl));
    }

    template <typename PM, must_be_member_function_pointer<PM> = 0>
    static cpp_function read(PM pm, const handle &hdl) {
        return readonly(pm, hdl);
    }

    template <typename PM, must_be_member_function_pointer<PM> = 0>
    static cpp_function write(PM pm, const handle &hdl) {
        return cpp_function([pm](T &c, const D &value) { c.*pm = value; }, is_method(hdl));
    }
};

PYBIND11_NAMESPACE_END(detail)

template <typename T, typename D, typename SFINAE = void>
struct property_cpp_function : detail::property_cpp_function_classic<T, D> {};

PYBIND11_NAMESPACE_BEGIN(detail)

template <typename T, typename D, typename SFINAE = void>
struct both_t_and_d_use_type_caster_base : std::false_type {};

// `T` is assumed to be equivalent to `intrinsic_t<T>`.
// `D` is may or may not be equivalent to `intrinsic_t<D>`.
template <typename T, typename D>
struct both_t_and_d_use_type_caster_base<
    T,
    D,
    enable_if_t<all_of<std::is_base_of<type_caster_base<T>, type_caster<T>>,
                       std::is_base_of<type_caster_base<intrinsic_t<D>>, make_caster<D>>>::value>>
    : std::true_type {};

// Specialization for raw pointer members, using smart_holder if that is the class_ holder,
// or falling back to the classic implementation if not.
// WARNING: Like the classic implementation, this implementation can lead to dangling pointers.
// See test_ptr() in tests/test_class_sh_property.py
// However, the read functions return a shared_ptr to the member, emulating the PyCLIF approach:
// https://github.com/google/clif/blob/c371a6d4b28d25d53a16e6d2a6d97305fb1be25a/clif/python/instance.h#L233
// This prevents disowning of the Python object owning the raw pointer member.
template <typename T, typename D>
struct property_cpp_function_sh_raw_ptr_member {
    using drp = typename std::remove_pointer<D>::type;

    template <typename PM, must_be_member_function_pointer<PM> = 0>
    static cpp_function readonly(PM pm, const handle &hdl) {
        type_info *tinfo = get_type_info(typeid(T), /*throw_if_missing=*/true);
        if (tinfo->holder_enum_v == holder_enum_t::smart_holder) {
            return cpp_function(
                [pm](handle c_hdl) -> std::shared_ptr<drp> {
                    std::shared_ptr<T> c_sp
                        = type_caster<std::shared_ptr<T>>::shared_ptr_with_responsible_parent(
                            c_hdl);
                    D ptr = (*c_sp).*pm;
                    return std::shared_ptr<drp>(c_sp, ptr);
                },
                is_method(hdl));
        }
        return property_cpp_function_classic<T, D>::readonly(pm, hdl);
    }

    template <typename PM, must_be_member_function_pointer<PM> = 0>
    static cpp_function read(PM pm, const handle &hdl) {
        return readonly(pm, hdl);
    }

    template <typename PM, must_be_member_function_pointer<PM> = 0>
    static cpp_function write(PM pm, const handle &hdl) {
        type_info *tinfo = get_type_info(typeid(T), /*throw_if_missing=*/true);
        if (tinfo->holder_enum_v == holder_enum_t::smart_holder) {
            return cpp_function([pm](T &c, D value) { c.*pm = std::forward<D>(std::move(value)); },
                                is_method(hdl));
        }
        return property_cpp_function_classic<T, D>::write(pm, hdl);
    }
};

// Specialization for members held by-value, using smart_holder if that is the class_ holder,
// or falling back to the classic implementation if not.
// The read functions return a shared_ptr to the member, emulating the PyCLIF approach:
// https://github.com/google/clif/blob/c371a6d4b28d25d53a16e6d2a6d97305fb1be25a/clif/python/instance.h#L233
// This prevents disowning of the Python object owning the member.
template <typename T, typename D>
struct property_cpp_function_sh_member_held_by_value {
    template <typename PM, must_be_member_function_pointer<PM> = 0>
    static cpp_function readonly(PM pm, const handle &hdl) {
        type_info *tinfo = get_type_info(typeid(T), /*throw_if_missing=*/true);
        if (tinfo->holder_enum_v == holder_enum_t::smart_holder) {
            return cpp_function(
                [pm](handle c_hdl) -> std::shared_ptr<typename std::add_const<D>::type> {
                    std::shared_ptr<T> c_sp
                        = type_caster<std::shared_ptr<T>>::shared_ptr_with_responsible_parent(
                            c_hdl);
                    return std::shared_ptr<typename std::add_const<D>::type>(c_sp,
                                                                             &(c_sp.get()->*pm));
                },
                is_method(hdl));
        }
        return property_cpp_function_classic<T, D>::readonly(pm, hdl);
    }

    template <typename PM, must_be_member_function_pointer<PM> = 0>
    static cpp_function read(PM pm, const handle &hdl) {
        type_info *tinfo = get_type_info(typeid(T), /*throw_if_missing=*/true);
        if (tinfo->holder_enum_v == holder_enum_t::smart_holder) {
            return cpp_function(
                [pm](handle c_hdl) -> std::shared_ptr<D> {
                    std::shared_ptr<T> c_sp
                        = type_caster<std::shared_ptr<T>>::shared_ptr_with_responsible_parent(
                            c_hdl);
                    return std::shared_ptr<D>(c_sp, &(c_sp.get()->*pm));
                },
                is_method(hdl));
        }
        return property_cpp_function_classic<T, D>::read(pm, hdl);
    }

    template <typename PM, must_be_member_function_pointer<PM> = 0>
    static cpp_function write(PM pm, const handle &hdl) {
        type_info *tinfo = get_type_info(typeid(T), /*throw_if_missing=*/true);
        if (tinfo->holder_enum_v == holder_enum_t::smart_holder) {
            return cpp_function([pm](T &c, const D &value) { c.*pm = value; }, is_method(hdl));
        }
        return property_cpp_function_classic<T, D>::write(pm, hdl);
    }
};

// Specialization for std::unique_ptr members, using smart_holder if that is the class_ holder,
// or falling back to the classic implementation if not.
// read disowns the member unique_ptr.
// write disowns the passed Python object.
// readonly is disabled (static_assert) because there is no safe & intuitive way to make the member
// accessible as a Python object without disowning the member unique_ptr. A .def_readonly disowning
// the unique_ptr member is deemed highly prone to misunderstandings.
template <typename T, typename D>
struct property_cpp_function_sh_unique_ptr_member {
    template <typename PM, must_be_member_function_pointer<PM> = 0>
    static cpp_function readonly(PM, const handle &) {
        static_assert(!is_instantiation<std::unique_ptr, D>::value,
                      "def_readonly cannot be used for std::unique_ptr members.");
        return cpp_function{}; // Unreachable.
    }

    template <typename PM, must_be_member_function_pointer<PM> = 0>
    static cpp_function read(PM pm, const handle &hdl) {
        type_info *tinfo = get_type_info(typeid(T), /*throw_if_missing=*/true);
        if (tinfo->holder_enum_v == holder_enum_t::smart_holder) {
            return cpp_function(
                [pm](handle c_hdl) -> D {
                    std::shared_ptr<T> c_sp
                        = type_caster<std::shared_ptr<T>>::shared_ptr_with_responsible_parent(
                            c_hdl);
                    return D{std::move(c_sp.get()->*pm)};
                },
                is_method(hdl));
        }
        return property_cpp_function_classic<T, D>::read(pm, hdl);
    }

    template <typename PM, must_be_member_function_pointer<PM> = 0>
    static cpp_function write(PM pm, const handle &hdl) {
        return cpp_function([pm](T &c, D &&value) { c.*pm = std::move(value); }, is_method(hdl));
    }
};

PYBIND11_NAMESPACE_END(detail)

template <typename T, typename D>
struct property_cpp_function<
    T,
    D,
    detail::enable_if_t<detail::all_of<std::is_pointer<D>,
                                       detail::both_t_and_d_use_type_caster_base<T, D>>::value>>
    : detail::property_cpp_function_sh_raw_ptr_member<T, D> {};

template <typename T, typename D>
struct property_cpp_function<T,
                             D,
                             detail::enable_if_t<detail::all_of<
                                 detail::none_of<std::is_pointer<D>,
                                                 std::is_array<D>,
                                                 detail::is_instantiation<std::unique_ptr, D>,
                                                 detail::is_instantiation<std::shared_ptr, D>>,
                                 detail::both_t_and_d_use_type_caster_base<T, D>>::value>>
    : detail::property_cpp_function_sh_member_held_by_value<T, D> {};

template <typename T, typename D>
struct property_cpp_function<
    T,
    D,
    detail::enable_if_t<detail::all_of<
        detail::is_instantiation<std::unique_ptr, D>,
        detail::both_t_and_d_use_type_caster_base<T, typename D::element_type>>::value>>
    : detail::property_cpp_function_sh_unique_ptr_member<T, D> {};

#ifdef PYBIND11_RUN_TESTING_WITH_SMART_HOLDER_AS_DEFAULT_BUT_NEVER_USE_IN_PRODUCTION_PLEASE
// NOTE: THIS IS MEANT FOR STRESS-TESTING OR TRIAGING ONLY!
//       Running the pybind11 unit tests with smart_holder as the default holder is to ensure
//       that `py::smart_holder` / `py::classh` is backward-compatible with all pre-existing
//       functionality.
//       Be careful not to link translation units compiled with different default holders, because
//       this will cause ODR violations (https://en.wikipedia.org/wiki/One_Definition_Rule).
template <typename>
using default_holder_type = smart_holder;
#else
template <typename T>
using default_holder_type = std::unique_ptr<T>;
#endif

template <typename type_, typename... options>
class class_ : public detail::generic_type {
    template <typename T>
    using is_holder = detail::is_holder_type<type_, T>;
    template <typename T>
    using is_subtype = detail::is_strict_base_of<type_, T>;
    template <typename T>
    using is_base = detail::is_strict_base_of<T, type_>;
    // struct instead of using here to help MSVC:
    template <typename T>
    struct is_valid_class_option : detail::any_of<is_holder<T>, is_subtype<T>, is_base<T>> {};

public:
    using type = type_;
    using type_alias = detail::exactly_one_t<is_subtype, void, options...>;
    constexpr static bool has_alias = !std::is_void<type_alias>::value;
    using holder_type = detail::exactly_one_t<is_holder, default_holder_type<type>, options...>;

    static_assert(detail::all_of<is_valid_class_option<options>...>::value,
                  "Unknown/invalid class_ template parameters provided");

    static_assert(!has_alias || std::is_polymorphic<type>::value,
                  "Cannot use an alias class (aka trampoline) with a non-polymorphic type");

#ifndef PYBIND11_RUN_TESTING_WITH_SMART_HOLDER_AS_DEFAULT_BUT_NEVER_USE_IN_PRODUCTION_PLEASE
    static_assert(!has_alias || !detail::is_smart_holder<holder_type>::value
                      || std::is_base_of<trampoline_self_life_support, type_alias>::value,
                  "Alias class (aka trampoline) must inherit from"
                  " pybind11::trampoline_self_life_support if used in combination with"
                  " pybind11::smart_holder");
#endif
    static_assert(!has_alias || detail::is_smart_holder<holder_type>::value
                      || !std::is_base_of<trampoline_self_life_support, type_alias>::value,
                  "pybind11::trampoline_self_life_support is a smart_holder feature, therefore"
                  " an alias class (aka trampoline) should inherit from"
                  " pybind11::trampoline_self_life_support only if used in combination with"
                  " pybind11::smart_holder");

    PYBIND11_OBJECT(class_, generic_type, PyType_Check)

    template <typename... Extra>
    class_(handle scope, const char *name, const Extra &...extra) {
        using namespace detail;

        // MI can only be specified via class_ template options, not constructor parameters
        static_assert(
            none_of<is_pyobject<Extra>...>::value || // no base class arguments, or:
                (constexpr_sum(is_pyobject<Extra>::value...) == 1 && // Exactly one base
                 constexpr_sum(is_base<options>::value...) == 0 &&   // no template option bases
                 // no multiple_inheritance attr
                 none_of<std::is_same<multiple_inheritance, Extra>...>::value),
            "Error: multiple inheritance bases must be specified via class_ template options");

        type_record record;
        record.scope = scope;
        record.name = name;
        record.type = &typeid(type);
        record.type_size = sizeof(conditional_t<has_alias, type_alias, type>);
        record.type_align = alignof(conditional_t<has_alias, type_alias, type> &);
        record.holder_size = sizeof(holder_type);
        record.init_instance = init_instance;

        if (detail::is_instantiation<std::unique_ptr, holder_type>::value) {
            record.holder_enum_v = detail::holder_enum_t::std_unique_ptr;
        } else if (detail::is_instantiation<std::shared_ptr, holder_type>::value) {
            record.holder_enum_v = detail::holder_enum_t::std_shared_ptr;
        } else if (std::is_same<holder_type, smart_holder>::value) {
            record.holder_enum_v = detail::holder_enum_t::smart_holder;
        } else {
            record.holder_enum_v = detail::holder_enum_t::custom_holder;
        }

        set_operator_new<type>(&record);

        /* Register base classes specified via template arguments to class_, if any */
        PYBIND11_EXPAND_SIDE_EFFECTS(add_base<options>(record));

        /* Process optional arguments, if any */
        process_attributes<Extra...>::init(extra..., &record);

        if (record.release_gil_before_calling_cpp_dtor) {
            record.dealloc = dealloc_release_gil_before_calling_cpp_dtor;
        } else {
            record.dealloc = dealloc_without_manipulating_gil;
        }

        if (std::is_base_of<trampoline_self_life_support, type_alias>::value) {
            // Store a cross-DSO-safe getter.
            // This lambda is defined in the same DSO that instantiates
            // class_<type, alias_type>, but it can be called safely from any other DSO.
            record.get_trampoline_self_life_support = [](void *type_ptr) {
                return dynamic_raw_ptr_cast_if_possible<trampoline_self_life_support>(
                    static_cast<type *>(type_ptr));
            };
        }

        generic_type::initialize(record);

        if (has_alias) {
            with_internals([&](internals &internals) {
                auto &local_internals = get_local_internals();
                if (record.module_local) {
                    local_internals.registered_types_cpp[&typeid(type_alias)]
                        = local_internals.registered_types_cpp[&typeid(type)];
                } else {
                    type_info *const val
                        = internals.registered_types_cpp[std::type_index(typeid(type))];
                    internals.registered_types_cpp[std::type_index(typeid(type_alias))] = val;
#if PYBIND11_INTERNALS_VERSION >= 12
                    internals.registered_types_cpp_fast[&typeid(type_alias)] = val;
#endif
                }
            });
        }
        def("_pybind11_conduit_v1_", cpp_conduit_method);
    }

    template <typename Base, detail::enable_if_t<is_base<Base>::value, int> = 0>
    static void add_base(detail::type_record &rec) {
        rec.add_base(typeid(Base), [](void *src) -> void * {
            return static_cast<Base *>(reinterpret_cast<type *>(src));
        });
        // Virtual inheritance means the base subobject is at a dynamic offset,
        // so the reinterpret_cast shortcut in load_impl Case 2a is invalid.
        // Force the MI path (implicit_casts) for correct pointer adjustment.
        // Detection: static_cast<Derived*>(Base*) is ill-formed for virtual bases.
        if PYBIND11_MAYBE_CONSTEXPR (!detail::is_static_downcastable<Base, type>::value) {
            rec.multiple_inheritance = true;
        }
    }

    template <typename Base, detail::enable_if_t<!is_base<Base>::value, int> = 0>
    static void add_base(detail::type_record &) {}

    template <typename Func, typename... Extra>
    PYBIND11_ALWAYS_INLINE class_ &def(const char *name_, Func &&f, const Extra &...extra) {
        cpp_function cf(method_adaptor<type>(std::forward<Func>(f)),
                        name(name_),
                        is_method(*this),
                        sibling(getattr(*this, name_, none())),
                        extra...);
        add_class_method(*this, name_, cf);
        return *this;
    }

    template <typename Func, typename... Extra>
    class_ &def_static(const char *name_, Func &&f, const Extra &...extra) {
        static_assert(!std::is_member_function_pointer<Func>::value,
                      "def_static(...) called with a non-static member function pointer");
        cpp_function cf(std::forward<Func>(f),
                        name(name_),
                        scope(*this),
                        sibling(getattr(*this, name_, none())),
                        extra...);
        auto cf_name = cf.name();
        attr(std::move(cf_name)) = staticmethod(std::move(cf));
        return *this;
    }

    template <typename T, typename... Extra, detail::enable_if_t<T::op_enable_if_hook, int> = 0>
    class_ &def(const T &op, const Extra &...extra) {
        op.execute(*this, extra...);
        return *this;
    }

    template <typename T, typename... Extra, detail::enable_if_t<T::op_enable_if_hook, int> = 0>
    class_ &def_cast(const T &op, const Extra &...extra) {
        op.execute_cast(*this, extra...);
        return *this;
    }

    template <typename... Args, typename... Extra>
    class_ &def(const detail::initimpl::constructor<Args...> &init, const Extra &...extra) {
        PYBIND11_WORKAROUND_INCORRECT_MSVC_C4100(init);
        init.execute(*this, extra...);
        return *this;
    }

    template <typename... Args, typename... Extra>
    class_ &def(const detail::initimpl::alias_constructor<Args...> &init, const Extra &...extra) {
        PYBIND11_WORKAROUND_INCORRECT_MSVC_C4100(init);
        init.execute(*this, extra...);
        return *this;
    }

    template <typename... Args, typename... Extra>
    class_ &def(detail::initimpl::factory<Args...> &&init, const Extra &...extra) {
        std::move(init).execute(*this, extra...);
        return *this;
    }

    template <typename... Args, typename... Extra>
    class_ &def(detail::initimpl::pickle_factory<Args...> &&pf, const Extra &...extra) {
        std::move(pf).execute(*this, extra...);
        return *this;
    }

    template <typename Func>
    class_ &def_buffer(Func &&func) {
        struct capture {
            Func func;
        };
        auto *ptr = new capture{std::forward<Func>(func)};
        install_buffer_funcs(
            [](PyObject *obj, void *ptr) -> buffer_info * {
                detail::make_caster<type> caster;
                if (!caster.load(obj, false)) {
                    return nullptr;
                }
                return new buffer_info(((capture *) ptr)->func(std::move(caster)));
            },
            ptr);
        weakref(m_ptr, cpp_function([ptr](handle wr) {
                    delete ptr;
                    wr.dec_ref();
                }))
            .release();
        return *this;
    }

    template <typename Return, typename Class, typename... Args>
    class_ &def_buffer(Return (Class::*func)(Args...)) {
        return def_buffer([func](type &obj) { return (obj.*func)(); });
    }

    template <typename Return, typename Class, typename... Args>
    class_ &def_buffer(Return (Class::*func)(Args...) const) {
        return def_buffer([func](const type &obj) { return (obj.*func)(); });
    }

    // Intentionally no &&/const&& overloads: buffer protocol callbacks are invoked on an
    // existing Python object and should not move-from self.
    template <typename Return, typename Class, typename... Args>
    class_ &def_buffer(Return (Class::*func)(Args...) &) {
        return def_buffer([func](type &obj) { return (obj.*func)(); });
    }

    template <typename Return, typename Class, typename... Args>
    class_ &def_buffer(Return (Class::*func)(Args...) const &) {
        return def_buffer([func](const type &obj) { return (obj.*func)(); });
    }

#ifdef __cpp_noexcept_function_type
    template <typename Return, typename Class, typename... Args>
    class_ &def_buffer(Return (Class::*func)(Args...) noexcept) {
        return def_buffer([func](type &obj) { return (obj.*func)(); });
    }

    template <typename Return, typename Class, typename... Args>
    class_ &def_buffer(Return (Class::*func)(Args...) const noexcept) {
        return def_buffer([func](const type &obj) { return (obj.*func)(); });
    }

    template <typename Return, typename Class, typename... Args>
    class_ &def_buffer(Return (Class::*func)(Args...) & noexcept) {
        return def_buffer([func](type &obj) { return (obj.*func)(); });
    }

    template <typename Return, typename Class, typename... Args>
    class_ &def_buffer(Return (Class::*func)(Args...) const & noexcept) {
        return def_buffer([func](const type &obj) { return (obj.*func)(); });
    }
#endif

    template <typename C, typename D, typename... Extra>
    class_ &def_readwrite(const char *name, D C::*pm, const Extra &...extra) {
        static_assert(std::is_same<C, type>::value || std::is_base_of<C, type>::value,
                      "def_readwrite() requires a class member (or base class member)");
        def_property(name,
                     property_cpp_function<type, D>::read(pm, *this),
                     property_cpp_function<type, D>::write(pm, *this),
                     return_value_policy::reference_internal,
                     extra...);
        return *this;
    }

    template <typename C, typename D, typename... Extra>
    class_ &def_readonly(const char *name, const D C::*pm, const Extra &...extra) {
        static_assert(std::is_same<C, type>::value || std::is_base_of<C, type>::value,
                      "def_readonly() requires a class member (or base class member)");
        def_property_readonly(name,
                              property_cpp_function<type, D>::readonly(pm, *this),
                              return_value_policy::reference_internal,
                              extra...);
        return *this;
    }

    template <typename D, typename... Extra>
    class_ &def_readwrite_static(const char *name, D *pm, const Extra &...extra) {
        cpp_function fget([pm](const object &) -> const D & { return *pm; }, scope(*this)),
            fset([pm](const object &, const D &value) { *pm = value; }, scope(*this));
        def_property_static(name, fget, fset, return_value_policy::reference, extra...);
        return *this;
    }

    template <typename D, typename... Extra>
    class_ &def_readonly_static(const char *name, const D *pm, const Extra &...extra) {
        cpp_function fget([pm](const object &) -> const D & { return *pm; }, scope(*this));
        def_property_readonly_static(name, fget, return_value_policy::reference, extra...);
        return *this;
    }

    /// Uses return_value_policy::reference_internal by default
    template <typename Getter, typename... Extra>
    class_ &def_property_readonly(const char *name, const Getter &fget, const Extra &...extra) {
        return def_property_readonly(name,
                                     cpp_function(method_adaptor<type>(fget)),
                                     return_value_policy::reference_internal,
                                     extra...);
    }

    /// Uses cpp_function's return_value_policy by default
    template <typename... Extra>
    class_ &
    def_property_readonly(const char *name, const cpp_function &fget, const Extra &...extra) {
        return def_property(name, fget, nullptr, extra...);
    }

    /// Uses return_value_policy::reference by default
    template <typename Getter, typename... Extra>
    class_ &
    def_property_readonly_static(const char *name, const Getter &fget, const Extra &...extra) {
        return def_property_readonly_static(
            name, cpp_function(fget), return_value_policy::reference, extra...);
    }

    /// Uses cpp_function's return_value_policy by default
    template <typename... Extra>
    class_ &def_property_readonly_static(const char *name,
                                         const cpp_function &fget,
                                         const Extra &...extra) {
        return def_property_static(name, fget, nullptr, extra...);
    }

    /// Uses return_value_policy::reference_internal by default
    template <typename Getter, typename Setter, typename... Extra>
    class_ &
    def_property(const char *name, const Getter &fget, const Setter &fset, const Extra &...extra) {
        return def_property(
            name, fget, cpp_function(method_adaptor<type>(fset), is_setter()), extra...);
    }
    template <typename Getter, typename... Extra>
    class_ &def_property(const char *name,
                         const Getter &fget,
                         const cpp_function &fset,
                         const Extra &...extra) {
        return def_property(name,
                            cpp_function(method_adaptor<type>(fget)),
                            fset,
                            return_value_policy::reference_internal,
                            extra...);
    }

    /// Uses cpp_function's return_value_policy by default
    template <typename... Extra>
    class_ &def_property(const char *name,
                         const cpp_function &fget,
                         const cpp_function &fset,
                         const Extra &...extra) {
        return def_property_static(name, fget, fset, is_method(*this), extra...);
    }

    /// Uses return_value_policy::reference by default
    template <typename Getter, typename... Extra>
    class_ &def_property_static(const char *name,
                                const Getter &fget,
                                const cpp_function &fset,
                                const Extra &...extra) {
        return def_property_static(
            name, cpp_function(fget), fset, return_value_policy::reference, extra...);
    }

    /// Uses cpp_function's return_value_policy by default
    template <typename... Extra>
    class_ &def_property_static(const char *name,
                                const cpp_function &fget,
                                const cpp_function &fset,
                                const Extra &...extra) {
        static_assert(0 == detail::constexpr_sum(std::is_base_of<arg, Extra>::value...),
                      "Argument annotations are not allowed for properties");
        static_assert(0 == detail::constexpr_sum(detail::is_call_guard<Extra>::value...),
                      "def_property family does not currently support call_guard. Use a "
                      "py::cpp_function instead.");
        static_assert(0 == detail::constexpr_sum(detail::is_keep_alive<Extra>::value...),
                      "def_property family does not currently support keep_alive. Use a "
                      "py::cpp_function instead.");
        auto rec_fget = get_function_record(fget), rec_fset = get_function_record(fset);
        auto *rec_active = rec_fget;
        if (rec_fget) {
            char *doc_prev = rec_fget->doc; /* 'extra' field may include a property-specific
                                               documentation string */
            auto args_before = rec_fget->args.size();
            detail::process_attributes<Extra...>::init(extra..., rec_fget);
            if (rec_fget->doc && rec_fget->doc != doc_prev) {
                std::free(doc_prev);
                rec_fget->doc = PYBIND11_COMPAT_STRDUP(rec_fget->doc);
            }
            // Args added by process_attributes (e.g. "self" via is_method + pos_only/kw_only)
            // need their strings strdup'd: initialize_generic's strdup loop already ran during
            // cpp_function construction, so it won't process these late additions. Without this,
            // destruct() would call free() on string literals. See gh-5976.
            for (auto i = args_before; i < rec_fget->args.size(); ++i) {
                if (rec_fget->args[i].name) {
                    rec_fget->args[i].name = PYBIND11_COMPAT_STRDUP(rec_fget->args[i].name);
                }
                if (rec_fget->args[i].descr) {
                    rec_fget->args[i].descr = PYBIND11_COMPAT_STRDUP(rec_fget->args[i].descr);
                }
            }
        }
        if (rec_fset) {
            char *doc_prev = rec_fset->doc;
            auto args_before = rec_fset->args.size();
            detail::process_attributes<Extra...>::init(extra..., rec_fset);
            if (rec_fset->doc && rec_fset->doc != doc_prev) {
                std::free(doc_prev);
                rec_fset->doc = PYBIND11_COMPAT_STRDUP(rec_fset->doc);
            }
            for (auto i = args_before; i < rec_fset->args.size(); ++i) {
                if (rec_fset->args[i].name) {
                    rec_fset->args[i].name = PYBIND11_COMPAT_STRDUP(rec_fset->args[i].name);
                }
                if (rec_fset->args[i].descr) {
                    rec_fset->args[i].descr = PYBIND11_COMPAT_STRDUP(rec_fset->args[i].descr);
                }
            }
            if (!rec_active) {
                rec_active = rec_fset;
            }
        }
        def_property_static_impl(name, fget, fset, rec_active);
        return *this;
    }

private:
    /// Initialize holder object, variant 1: object derives from enable_shared_from_this
    template <typename T>
    static void init_holder(detail::instance *inst,
                            detail::value_and_holder &v_h,
                            const holder_type * /* unused */,
                            const std::enable_shared_from_this<T> * /* dummy */) {

        auto sh = std::dynamic_pointer_cast<typename holder_type::element_type>(
            detail::try_get_shared_from_this(v_h.value_ptr<type>()));
        if (sh) {
            new (std::addressof(v_h.holder<holder_type>())) holder_type(std::move(sh));
            v_h.set_holder_constructed();
        }

        if (!v_h.holder_constructed() && inst->owned) {
            new (std::addressof(v_h.holder<holder_type>())) holder_type(v_h.value_ptr<type>());
            v_h.set_holder_constructed();
        }
    }

    static void init_holder_from_existing(const detail::value_and_holder &v_h,
                                          const holder_type *holder_ptr,
                                          std::true_type /*is_copy_constructible*/) {
        new (std::addressof(v_h.holder<holder_type>())) holder_type(*holder_ptr);
    }

    static void init_holder_from_existing(const detail::value_and_holder &v_h,
                                          const holder_type *holder_ptr,
                                          std::false_type /*is_copy_constructible*/) {
        new (std::addressof(v_h.holder<holder_type>()))
            holder_type(std::move(*const_cast<holder_type *>(holder_ptr)));
    }

    /// Initialize holder object, variant 2: try to construct from existing holder object, if
    /// possible
    static void init_holder(detail::instance *inst,
                            detail::value_and_holder &v_h,
                            const holder_type *holder_ptr,
                            const void * /* dummy -- not enable_shared_from_this<T>) */) {
        if (holder_ptr) {
            init_holder_from_existing(v_h, holder_ptr, std::is_copy_constructible<holder_type>());
            v_h.set_holder_constructed();
        } else if (detail::always_construct_holder<holder_type>::value || inst->owned) {
            new (std::addressof(v_h.holder<holder_type>())) holder_type(v_h.value_ptr<type>());
            v_h.set_holder_constructed();
        }
    }

    /// Performs instance initialization including constructing a holder and registering the known
    /// instance.  Should be called as soon as the `type` value_ptr is set for an instance.  Takes
    /// an optional pointer to an existing holder to use; if not specified and the instance is
    /// `.owned`, a new holder will be constructed to manage the value pointer.
    template <typename H = holder_type,
              detail::enable_if_t<!detail::is_smart_holder<H>::value, int> = 0>
    static void init_instance(detail::instance *inst, const void *holder_ptr) {
        auto v_h = inst->get_value_and_holder(detail::get_type_info(typeid(type)));
        if (!v_h.instance_registered()) {
            register_instance(inst, v_h.value_ptr(), v_h.type);
            v_h.set_instance_registered();
        }
        init_holder(inst, v_h, (const holder_type *) holder_ptr, v_h.value_ptr<type>());
    }

    template <typename WrappedType>
    static bool try_initialization_using_shared_from_this(holder_type *, WrappedType *, ...) {
        return false;
    }

    // Adopting existing approach used by type_caster_base, although it leads to somewhat fuzzy
    // ownership semantics: if we detected via shared_from_this that a shared_ptr exists already,
    // it is reused, irrespective of the return_value_policy in effect.
    // "SomeBaseOfWrappedType" is needed because std::enable_shared_from_this is not necessarily a
    // direct base of WrappedType.
    template <typename WrappedType, typename SomeBaseOfWrappedType>
    static bool try_initialization_using_shared_from_this(
        holder_type *uninitialized_location,
        WrappedType *value_ptr_w_t,
        const std::enable_shared_from_this<SomeBaseOfWrappedType> *) {
        auto shd_ptr = std::dynamic_pointer_cast<WrappedType>(
            detail::try_get_shared_from_this(value_ptr_w_t));
        if (!shd_ptr) {
            return false;
        }
        // Note: inst->owned ignored.
        new (uninitialized_location) holder_type(holder_type::from_shared_ptr(shd_ptr));
        return true;
    }

    template <typename H = holder_type,
              detail::enable_if_t<detail::is_smart_holder<H>::value, int> = 0>
    static void init_instance(detail::instance *inst, const void *holder_const_void_ptr) {
        // Need for const_cast is a consequence of the type_info::init_instance type:
        // void (*init_instance)(instance *, const void *);
        auto *holder_void_ptr = const_cast<void *>(holder_const_void_ptr);

        auto v_h = inst->get_value_and_holder(detail::get_type_info(typeid(type)));
        if (!v_h.instance_registered()) {
            register_instance(inst, v_h.value_ptr(), v_h.type);
            v_h.set_instance_registered();
        }
        auto *uninitialized_location = std::addressof(v_h.holder<holder_type>());
        auto *value_ptr_w_t = v_h.value_ptr<type>();
        // Try downcast from `type` to `type_alias`:
        inst->is_alias
            = detail::dynamic_raw_ptr_cast_if_possible<type_alias>(value_ptr_w_t) != nullptr;
        if (holder_void_ptr) {
            // Note: inst->owned ignored.
            auto *holder_ptr = static_cast<holder_type *>(holder_void_ptr);
            new (uninitialized_location) holder_type(std::move(*holder_ptr));
        } else if (!try_initialization_using_shared_from_this(
                       uninitialized_location, value_ptr_w_t, value_ptr_w_t)) {
            if (inst->owned) {
                new (uninitialized_location) holder_type(holder_type::from_raw_ptr_take_ownership(
                    value_ptr_w_t, /*void_cast_raw_ptr*/ inst->is_alias));
            } else {
                new (uninitialized_location)
                    holder_type(holder_type::from_raw_ptr_unowned(value_ptr_w_t));
            }
        }
        v_h.set_holder_constructed();
    }

    // Deallocates an instance; via holder, if constructed; otherwise via operator delete.
    // NOTE: The Python error indicator needs to cleared BEFORE this function is called.
    // This is because we could be deallocating while cleaning up after a Python exception.
    // If the error indicator is not cleared but the C++ destructor code makes Python C API
    // calls, those calls are likely to generate a new exception, and pybind11 will then
    // throw `error_already_set` from the C++ destructor. This is forbidden and will
    // trigger std::terminate().
    static void dealloc_impl(detail::value_and_holder &v_h) {
        if (v_h.holder_constructed()) {
            v_h.holder<holder_type>().~holder_type();
            v_h.set_holder_constructed(false);
        } else {
            detail::call_operator_delete(
                v_h.value_ptr<type>(), v_h.type->type_size, v_h.type->type_align);
        }
        v_h.value_ptr() = nullptr;
    }

    static void dealloc_without_manipulating_gil(detail::value_and_holder &v_h) {
        error_scope scope;
        dealloc_impl(v_h);
    }

    static void dealloc_release_gil_before_calling_cpp_dtor(detail::value_and_holder &v_h) {
        error_scope scope;
        // Intentionally not using `gil_scoped_release` because the non-simple
        // version unconditionally calls `get_internals()`.
        // `Py_BEGIN_ALLOW_THREADS`, `Py_END_ALLOW_THREADS` cannot be used
        // because those macros include `{` and `}`.
        PyThreadState *py_ts = PyEval_SaveThread();
        try {
            dealloc_impl(v_h);
        } catch (...) {
            // This code path is expected to be unreachable unless there is a
            // bug in pybind11 itself.
            // An alternative would be to mark this function, or
            // `dealloc_impl()`, with `nothrow`, but that would be a subtle
            // behavior change and could make debugging more difficult.
            PyEval_RestoreThread(py_ts);
            throw;
        }
        PyEval_RestoreThread(py_ts);
    }

    static detail::function_record *get_function_record(handle h) {
        h = detail::get_function(h);
        if (!h) {
            return nullptr;
        }

        handle func_self = PyCFunction_GET_SELF(h.ptr());
        if (!func_self) {
            throw error_already_set();
        }
        return detail::function_record_ptr_from_PyObject(func_self.ptr());
    }
};

// Supports easier switching between py::class_<T> and py::class_<T, py::smart_holder>:
// users can simply replace the `_` in `class_` with `h` or vice versa.
template <typename type_, typename... options>
using classh = class_<type_, smart_holder, options...>;

/// Binds an existing constructor taking arguments Args...
template <typename... Args>
detail::initimpl::constructor<Args...> init() {
    return {};
}
/// Like `init<Args...>()`, but the instance is always constructed through the alias class (even
/// when not inheriting on the Python side).
template <typename... Args>
detail::initimpl::alias_constructor<Args...> init_alias() {
    return {};
}

/// Binds a factory function as a constructor
template <typename Func, typename Ret = detail::initimpl::factory<Func>>
Ret init(Func &&f) {
    return {std::forward<Func>(f)};
}

/// Dual-argument factory function: the first function is called when no alias is needed, the
/// second when an alias is needed (i.e. due to python-side inheritance).  Arguments must be
/// identical.
template <typename CFunc, typename AFunc, typename Ret = detail::initimpl::factory<CFunc, AFunc>>
Ret init(CFunc &&c, AFunc &&a) {
    return {std::forward<CFunc>(c), std::forward<AFunc>(a)};
}

/// Binds pickling functions `__getstate__` and `__setstate__` and ensures that the type
/// returned by `__getstate__` is the same as the argument accepted by `__setstate__`.
template <typename GetState, typename SetState>
detail::initimpl::pickle_factory<GetState, SetState> pickle(GetState &&g, SetState &&s) {
    return {std::forward<GetState>(g), std::forward<SetState>(s)};
}

PYBIND11_NAMESPACE_BEGIN(detail)

str enum_name(handle arg);

struct enum_base {
    enum_base(const handle &base, const handle &parent) : m_base(base), m_parent(parent) {}

    void init(bool is_arithmetic, bool is_convertible);

    void value(char const *name_, object value, const char *doc = nullptr);

    void export_values();

    handle m_base;
    handle m_parent;
};

template <bool is_signed, size_t length>
struct equivalent_integer {};
template <>
struct equivalent_integer<true, 1> {
    using type = int8_t;
};
template <>
struct equivalent_integer<false, 1> {
    using type = uint8_t;
};
template <>
struct equivalent_integer<true, 2> {
    using type = int16_t;
};
template <>
struct equivalent_integer<false, 2> {
    using type = uint16_t;
};
template <>
struct equivalent_integer<true, 4> {
    using type = int32_t;
};
template <>
struct equivalent_integer<false, 4> {
    using type = uint32_t;
};
template <>
struct equivalent_integer<true, 8> {
    using type = int64_t;
};
template <>
struct equivalent_integer<false, 8> {
    using type = uint64_t;
};

template <typename IntLike>
using equivalent_integer_t =
    typename equivalent_integer<std::is_signed<IntLike>::value, sizeof(IntLike)>::type;

PYBIND11_NAMESPACE_END(detail)

/// Binds C++ enumerations and enumeration classes to Python
template <typename Type>
class enum_ : public class_<Type> {
public:
    using Base = class_<Type>;
    using Base::attr;
    using Base::def_property_readonly;
    using Base::def_property_readonly_static;
    using Underlying = typename std::underlying_type<Type>::type;
    // Scalar is the integer representation of underlying type
    using Scalar = detail::conditional_t<detail::any_of<detail::is_std_char_type<Underlying>,
                                                        std::is_same<Underlying, bool>>::value,
                                         detail::equivalent_integer_t<Underlying>,
                                         Underlying>;

    template <typename... Extra>
    enum_(const handle &scope, const char *name, const Extra &...extra)
        : class_<Type>(scope, name, extra...), m_base(*this, scope) {
        {
            if (detail::global_internals_native_enum_type_map_contains(
                    std::type_index(typeid(Type)))) {
                pybind11_fail("pybind11::enum_ \"" + std::string(name)
                              + "\" is already registered as a pybind11::native_enum!");
            }
        }

        constexpr bool is_arithmetic = detail::any_of<std::is_same<arithmetic, Extra>...>::value;
        constexpr bool is_convertible = std::is_convertible<Type, Underlying>::value;
        m_base.init(is_arithmetic, is_convertible);

        def(init([](Scalar i) { return static_cast<Type>(i); }), arg("value"));
        def_property_readonly("value", [](Type value) { return (Scalar) value; }, pos_only());
#define PYBIND11_ENUM_OP_SAME_TYPE(op, expr)                                                      \
    def(op, [](Type a, Type b) { return expr; }, pybind11::name(op), arg("other"), pos_only())
#define PYBIND11_ENUM_OP_SAME_TYPE_RHS_MAY_BE_NONE(op, expr)                                      \
    def(op, [](Type a, Type *b_ptr) { return expr; }, pybind11::name(op), arg("other"), pos_only())
#define PYBIND11_ENUM_OP_SCALAR(op, op_expr)                                                      \
    def(                                                                                          \
        op,                                                                                       \
        [](Type a, Scalar b) { return static_cast<Scalar>(a) op_expr b; },                        \
        pybind11::name(op),                                                                       \
        arg("other"),                                                                             \
        pos_only())
#define PYBIND11_ENUM_OP_CONV_ARITHMETIC(op, op_expr)                                             \
    /* NOLINTNEXTLINE(bugprone-macro-parentheses) */                                              \
    PYBIND11_ENUM_OP_SAME_TYPE(op, static_cast<Scalar>(a) op_expr static_cast<Scalar>(b));        \
    PYBIND11_ENUM_OP_SCALAR(op, op_expr)
#define PYBIND11_ENUM_OP_REJECT_UNRELATED_TYPE(op, strict_behavior)                               \
    def(                                                                                          \
        op,                                                                                       \
        [](Type, const object &) { strict_behavior; },                                            \
        pybind11::name(op),                                                                       \
        arg("other"),                                                                             \
        pos_only())
#define PYBIND11_ENUM_OP_STRICT_ARITHMETIC(op, op_expr, strict_behavior)                          \
    /* NOLINTNEXTLINE(bugprone-macro-parentheses) */                                              \
    PYBIND11_ENUM_OP_SAME_TYPE(op, static_cast<Scalar>(a) op_expr static_cast<Scalar>(b));        \
    PYBIND11_ENUM_OP_REJECT_UNRELATED_TYPE(op, strict_behavior);

        PYBIND11_ENUM_OP_SAME_TYPE_RHS_MAY_BE_NONE("__eq__", b_ptr && a == *b_ptr);
        PYBIND11_ENUM_OP_SAME_TYPE_RHS_MAY_BE_NONE("__ne__", !b_ptr || a != *b_ptr);
        if (std::is_convertible<Type, Scalar>::value) {
            PYBIND11_ENUM_OP_SCALAR("__eq__", ==);
            PYBIND11_ENUM_OP_SCALAR("__ne__", !=);
            if (is_arithmetic) {
                PYBIND11_ENUM_OP_CONV_ARITHMETIC("__lt__", <);
                PYBIND11_ENUM_OP_CONV_ARITHMETIC("__gt__", >);
                PYBIND11_ENUM_OP_CONV_ARITHMETIC("__le__", <=);
                PYBIND11_ENUM_OP_CONV_ARITHMETIC("__ge__", >=);
                PYBIND11_ENUM_OP_CONV_ARITHMETIC("__and__", &);
                PYBIND11_ENUM_OP_CONV_ARITHMETIC("__rand__", &);
                PYBIND11_ENUM_OP_CONV_ARITHMETIC("__or__", |);
                PYBIND11_ENUM_OP_CONV_ARITHMETIC("__ror__", |);
                PYBIND11_ENUM_OP_CONV_ARITHMETIC("__xor__", ^);
                PYBIND11_ENUM_OP_CONV_ARITHMETIC("__rxor__", ^);
            }
        } else if (is_arithmetic) {
#define PYBIND11_ENUM_OP_THROW_TYPE_ERROR                                                         \
    throw type_error("Expected an enumeration of matching type!");
            PYBIND11_ENUM_OP_STRICT_ARITHMETIC("__lt__", <, PYBIND11_ENUM_OP_THROW_TYPE_ERROR);
            PYBIND11_ENUM_OP_STRICT_ARITHMETIC("__gt__", >, PYBIND11_ENUM_OP_THROW_TYPE_ERROR);
            PYBIND11_ENUM_OP_STRICT_ARITHMETIC("__le__", <=, PYBIND11_ENUM_OP_THROW_TYPE_ERROR);
            PYBIND11_ENUM_OP_STRICT_ARITHMETIC("__ge__", >=, PYBIND11_ENUM_OP_THROW_TYPE_ERROR);
#undef PYBIND11_ENUM_OP_THROW_TYPE_ERROR
        }
        PYBIND11_ENUM_OP_REJECT_UNRELATED_TYPE("__eq__", return false);
        PYBIND11_ENUM_OP_REJECT_UNRELATED_TYPE("__ne__", return true);

#undef PYBIND11_ENUM_OP_SAME_TYPE
#undef PYBIND11_ENUM_OP_SAME_TYPE_RHS_MAY_BE_NONE
#undef PYBIND11_ENUM_OP_SCALAR
#undef PYBIND11_ENUM_OP_CONV_ARITHMETIC
#undef PYBIND11_ENUM_OP_REJECT_UNRELATED_TYPE
#undef PYBIND11_ENUM_OP_STRICT_ARITHMETIC

        def("__int__", [](Type value) { return (Scalar) value; }, pos_only());
        def("__index__", [](Type value) { return (Scalar) value; }, pos_only());
        attr("__setstate__") = cpp_function(
            [](detail::value_and_holder &v_h, Scalar arg) {
                detail::initimpl::setstate<Base>(
                    v_h, static_cast<Type>(arg), Py_TYPE(v_h.inst) != v_h.type->type);
            },
            detail::is_new_style_constructor(),
            pybind11::name("__setstate__"),
            is_method(*this),
            arg("state"),
            pos_only());
    }

    template <typename Func, typename... Extra>
    enum_ &def(const char *name_, Func &&f, const Extra &...extra) {
        if (std::strcmp(name_, "__str__") == 0) {
            Base::def(name_, std::forward<Func>(f), prepend{}, extra...);
        } else {
            Base::def(name_, std::forward<Func>(f), extra...);
        }
        return *this;
    }

    // Avoid using Base::def here: GCC 15/MinGW sees the duplicate dependent-base
    // def(const char *, ...) template as ambiguous with enum_::def(const char *, ...).
    template <typename T,
              typename... Extra,
              detail::enable_if_t<
                  !std::is_convertible<typename std::decay<T>::type, const char *>::value,
                  int> = 0>
    enum_ &def(T &&op, const Extra &...extra) {
        Base::def(std::forward<T>(op), extra...);
        return *this;
    }

    /// Export enumeration entries into the parent scope
    enum_ &export_values() {
        m_base.export_values();
        return *this;
    }

    /// Add an enumeration entry
    enum_ &value(char const *name, Type value, const char *doc = nullptr) {
        m_base.value(name, pybind11::cast(value, return_value_policy::copy), doc);
        return *this;
    }

private:
    detail::enum_base m_base;
};

PYBIND11_NAMESPACE_BEGIN(detail)

/* There are a large number of apparently unused template arguments because
 * each combination requires a separate py::class_ registration.
 */
template <typename Access,
          return_value_policy Policy,
          typename Iterator,
          typename Sentinel,
          typename ValueType,
          typename... Extra>
struct iterator_state {
    Iterator it;
    Sentinel end;
    bool first_or_done;
};

// Note: these helpers take the iterator by non-const reference because some
// iterators in the wild can't be dereferenced when const. The & after Iterator
// is required for MSVC < 16.9. SFINAE cannot be reused for result_type due to
// bugs in ICC, NVCC, and PGI compilers. See PR #3293.
template <typename Iterator, typename SFINAE = decltype(*std::declval<Iterator &>())>
struct iterator_access {
    using result_type = decltype(*std::declval<Iterator &>());
    // NOLINTNEXTLINE(readability-const-return-type) // PR #3263
    result_type operator()(Iterator &it) const { return *it; }
};

template <typename Iterator, typename SFINAE = decltype((*std::declval<Iterator &>()).first)>
class iterator_key_access {
private:
    using pair_type = decltype(*std::declval<Iterator &>());

public:
    /* If either the pair itself or the element of the pair is a reference, we
     * want to return a reference, otherwise a value. When the decltype
     * expression is parenthesized it is based on the value category of the
     * expression; otherwise it is the declared type of the pair member.
     * The use of declval<pair_type> in the second branch rather than directly
     * using *std::declval<Iterator &>() is a workaround for nvcc
     * (it's not used in the first branch because going via decltype and back
     * through declval does not perfectly preserve references).
     */
    using result_type
        = conditional_t<std::is_reference<decltype(*std::declval<Iterator &>())>::value,
                        decltype(((*std::declval<Iterator &>()).first)),
                        decltype(std::declval<pair_type>().first)>;
    result_type operator()(Iterator &it) const { return (*it).first; }
};

template <typename Iterator, typename SFINAE = decltype((*std::declval<Iterator &>()).second)>
class iterator_value_access {
private:
    using pair_type = decltype(*std::declval<Iterator &>());

public:
    using result_type
        = conditional_t<std::is_reference<decltype(*std::declval<Iterator &>())>::value,
                        decltype(((*std::declval<Iterator &>()).second)),
                        decltype(std::declval<pair_type>().second)>;
    result_type operator()(Iterator &it) const { return (*it).second; }
};

template <typename Access,
          return_value_policy Policy,
          typename Iterator,
          typename Sentinel,
          typename ValueType,
          typename... Extra>
// NOLINTNEXTLINE(performance-unnecessary-value-param)
iterator make_iterator_impl(Iterator first, Sentinel last, Extra &&...extra) {
    using state = detail::iterator_state<Access, Policy, Iterator, Sentinel, ValueType, Extra...>;
    // TODO: state captures only the types of Extra, not the values

    // For Python < 3.14.0rc1, pycritical_section uses direct mutex locking (same as a unique
    // lock), which may deadlock during type registration. See detail/internals.h for details.
#if PY_VERSION_HEX >= 0x030E00C1 // 3.14.0rc1
    PYBIND11_LOCK_INTERNALS(get_internals());
#endif
    if (!detail::get_type_info(typeid(state), false)) {
        class_<state>(handle(), "iterator", pybind11::module_local())
            .def(
                "__iter__", [](state &s) -> state & { return s; }, pos_only())
            .def(
                "__next__",
                [](state &s) -> ValueType {
                    if (!s.first_or_done) {
                        ++s.it;
                    } else {
                        s.first_or_done = false;
                    }
                    if (s.it == s.end) {
                        s.first_or_done = true;
                        throw stop_iteration();
                    }
                    return Access()(s.it);
                    // NOLINTNEXTLINE(readability-const-return-type) // PR #3263
                },
                std::forward<Extra>(extra)...,
                pos_only(),
                Policy);
    }

    return cast(state{std::forward<Iterator>(first), std::forward<Sentinel>(last), true});
}

PYBIND11_NAMESPACE_END(detail)

/// Makes a python iterator from a first and past-the-end C++ InputIterator.
template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Iterator,
          typename Sentinel,
          typename ValueType = typename detail::iterator_access<Iterator>::result_type,
          typename... Extra>
// NOLINTNEXTLINE(performance-unnecessary-value-param)
typing::Iterator<ValueType> make_iterator(Iterator first, Sentinel last, Extra &&...extra) {
    return detail::make_iterator_impl<detail::iterator_access<Iterator>,
                                      Policy,
                                      Iterator,
                                      Sentinel,
                                      ValueType,
                                      Extra...>(std::forward<Iterator>(first),
                                                std::forward<Sentinel>(last),
                                                std::forward<Extra>(extra)...);
}

/// Makes a python iterator over the keys (`.first`) of a iterator over pairs from a
/// first and past-the-end InputIterator.
template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Iterator,
          typename Sentinel,
          typename KeyType = typename detail::iterator_key_access<Iterator>::result_type,
          typename... Extra>
typing::Iterator<KeyType> make_key_iterator(Iterator first, Sentinel last, Extra &&...extra) {
    return detail::make_iterator_impl<detail::iterator_key_access<Iterator>,
                                      Policy,
                                      Iterator,
                                      Sentinel,
                                      KeyType,
                                      Extra...>(std::forward<Iterator>(first),
                                                std::forward<Sentinel>(last),
                                                std::forward<Extra>(extra)...);
}

/// Makes a python iterator over the values (`.second`) of a iterator over pairs from a
/// first and past-the-end InputIterator.
template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Iterator,
          typename Sentinel,
          typename ValueType = typename detail::iterator_value_access<Iterator>::result_type,
          typename... Extra>
typing::Iterator<ValueType> make_value_iterator(Iterator first, Sentinel last, Extra &&...extra) {
    return detail::make_iterator_impl<detail::iterator_value_access<Iterator>,
                                      Policy,
                                      Iterator,
                                      Sentinel,
                                      ValueType,
                                      Extra...>(std::forward<Iterator>(first),
                                                std::forward<Sentinel>(last),
                                                std::forward<Extra>(extra)...);
}

/// Makes an iterator over values of an stl container or other container supporting
/// `std::begin()`/`std::end()`
template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Type,
          typename ValueType = typename detail::iterator_access<
              decltype(std::begin(std::declval<Type &>()))>::result_type,
          typename... Extra>
typing::Iterator<ValueType> make_iterator(Type &value, Extra &&...extra) {
    return make_iterator<Policy>(
        std::begin(value), std::end(value), std::forward<Extra>(extra)...);
}

/// Makes an iterator over the keys (`.first`) of a stl map-like container supporting
/// `std::begin()`/`std::end()`
template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Type,
          typename KeyType = typename detail::iterator_key_access<
              decltype(std::begin(std::declval<Type &>()))>::result_type,
          typename... Extra>
typing::Iterator<KeyType> make_key_iterator(Type &value, Extra &&...extra) {
    return make_key_iterator<Policy>(
        std::begin(value), std::end(value), std::forward<Extra>(extra)...);
}

/// Makes an iterator over the values (`.second`) of a stl map-like container supporting
/// `std::begin()`/`std::end()`
template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Type,
          typename ValueType = typename detail::iterator_value_access<
              decltype(std::begin(std::declval<Type &>()))>::result_type,
          typename... Extra>
typing::Iterator<ValueType> make_value_iterator(Type &value, Extra &&...extra) {
    return make_value_iterator<Policy>(
        std::begin(value), std::end(value), std::forward<Extra>(extra)...);
}

template <typename InputType, typename OutputType>
void implicitly_convertible() {
    struct set_flag {
        bool &flag;
        explicit set_flag(bool &flag_) : flag(flag_) { flag_ = true; }
        ~set_flag() { flag = false; }

        // Prevent copying/moving to ensure RAII guard is used safely
        set_flag(const set_flag &) = delete;
        set_flag(set_flag &&) = delete;
        set_flag &operator=(const set_flag &) = delete;
        set_flag &operator=(set_flag &&) = delete;
    };
    auto implicit_caster = [](PyObject *obj, PyTypeObject *type) -> PyObject * {
        thread_local bool currently_used = false;
        if (currently_used) { // implicit conversions are non-reentrant
            return nullptr;
        }
        set_flag flag_helper(currently_used);
        if (!detail::make_caster<InputType>().load(obj, false)) {
            return nullptr;
        }
        tuple args(1);
        args[0] = obj;
        PyObject *result = PyObject_Call(reinterpret_cast<PyObject *>(type), args.ptr(), nullptr);
        if (result == nullptr) {
            PyErr_Clear();
        }
        return result;
    };

    if (auto *tinfo = detail::get_type_info(typeid(OutputType))) {
        tinfo->implicit_conversions.emplace_back(std::move(implicit_caster));
    } else {
        pybind11_fail("implicitly_convertible: Unable to find type " + type_id<OutputType>());
    }
}

void register_exception_translator(ExceptionTranslator &&translator);

/**
 * Add a new module-local exception translator. Locally registered functions
 * will be tried before any globally registered exception translators, which
 * will only be invoked if the module-local handlers do not deal with
 * the exception.
 */
void register_local_exception_translator(ExceptionTranslator &&translator);

/**
 * Wrapper to generate a new Python exception type.
 *
 * This should only be used with py::set_error() for now.
 * It is not (yet) possible to use as a py::base.
 * Template type argument is reserved for future use.
 */
template <typename type>
class exception : public object {
public:
    exception() = default;
    exception(handle scope, const char *name, handle base = PyExc_Exception) {
        std::string full_name
            = scope.attr("__name__").cast<std::string>() + std::string(".") + name;
        m_ptr = PyErr_NewException(const_cast<char *>(full_name.c_str()), base.ptr(), nullptr);
        if (hasattr(scope, "__dict__") && scope.attr("__dict__").contains(name)) {
            pybind11_fail("Error during initialization: multiple incompatible "
                          "definitions with name \""
                          + std::string(name) + "\"");
        }
        scope.attr(name) = *this;
    }

    // Sets the current python exception to this exception object with the given message
    PYBIND11_DEPRECATED("Please use py::set_error() instead "
                        "(https://github.com/pybind/pybind11/pull/4772)")
    void operator()(const char *message) const { set_error(*this, message); }
};

PYBIND11_NAMESPACE_BEGIN(detail)

template <>
struct handle_type_name<exception<void>> {
    static constexpr auto name = const_name("Exception");
};

// Helper function for register_exception and register_local_exception
template <typename CppException>
exception<CppException> &
register_exception_impl(handle scope, const char *name, handle base, bool isLocal) {
    PYBIND11_CONSTINIT static gil_safe_call_once_and_store<exception<CppException>> exc_storage;
    exc_storage.call_once_and_store_result(
        [&]() { return exception<CppException>(scope, name, base); });

    auto register_func
        = isLocal ? &register_local_exception_translator : &register_exception_translator;

    register_func([](std::exception_ptr p) {
        if (!p) {
            return;
        }
        try {
            std::rethrow_exception(p);
        } catch (const CppException &e) {
            set_error(exc_storage.get_stored(), e.what());
        }
    });
    return exc_storage.get_stored();
}

PYBIND11_NAMESPACE_END(detail)

/**
 * Registers a Python exception in `m` of the given `name` and installs a translator to
 * translate the C++ exception to the created Python exception using the what() method.
 * This is intended for simple exception translations; for more complex translation, register the
 * exception object and translator directly.
 */
template <typename CppException>
exception<CppException> &
register_exception(handle scope, const char *name, handle base = PyExc_Exception) {
    return detail::register_exception_impl<CppException>(scope, name, base, false /* isLocal */);
}

/**
 * Registers a Python exception in `m` of the given `name` and installs a translator to
 * translate the C++ exception to the created Python exception using the what() method.
 * This translator will only be used for exceptions that are thrown in this module and will be
 * tried before global exception translators, including those registered with register_exception.
 * This is intended for simple exception translations; for more complex translation, register the
 * exception object and translator directly.
 */
template <typename CppException>
exception<CppException> &
register_local_exception(handle scope, const char *name, handle base = PyExc_Exception) {
    return detail::register_exception_impl<CppException>(scope, name, base, true /* isLocal */);
}

PYBIND11_NAMESPACE_BEGIN(detail)
void print(const tuple &args, const dict &kwargs);
PYBIND11_NAMESPACE_END(detail)

template <return_value_policy policy = return_value_policy::automatic_reference, typename... Args>
void print(Args &&...args) {
    auto c = detail::collect_arguments<policy>(std::forward<Args>(args)...);
    detail::print(c.args(), c.kwargs());
}

// (error_already_set::m_fetched_error_deleter and ::what are declared in pytypes.h;
// definitions are in pybind11-inl.h.)

PYBIND11_NAMESPACE_BEGIN(detail)

function get_type_override(const void *this_ptr, const type_info *this_type, const char *name);
PYBIND11_NAMESPACE_END(detail)

/** \rst
  Try to retrieve a python method by the provided name from the instance pointed to by the
  this_ptr.

  :this_ptr: The pointer to the object the overridden method should be retrieved for. This should
             be the first non-trampoline class encountered in the inheritance chain.
  :name: The name of the overridden Python method to retrieve.
  :return: The Python method by this name from the object or an empty function wrapper.
 \endrst */
template <class T>
function get_override(const T *this_ptr, const char *name) {
    auto *tinfo = detail::get_type_info(typeid(T));
    return tinfo ? detail::get_type_override(this_ptr, tinfo, name) : function();
}

#define PYBIND11_OVERRIDE_IMPL(ret_type, cname, name, ...)                                        \
    do {                                                                                          \
        pybind11::gil_scoped_acquire gil;                                                         \
        pybind11::function override                                                               \
            = pybind11::get_override(static_cast<const cname *>(this), name);                     \
        if (override) {                                                                           \
            auto o = override(__VA_ARGS__);                                                       \
            PYBIND11_WARNING_PUSH                                                                 \
            PYBIND11_WARNING_DISABLE_MSVC(4127)                                                   \
            if PYBIND11_MAYBE_CONSTEXPR (                                                         \
                pybind11::detail::cast_is_temporary_value_reference<ret_type>::value              \
                && !pybind11::detail::is_same_ignoring_cvref<ret_type, PyObject *>::value) {      \
                static pybind11::detail::override_caster_t<ret_type> caster;                      \
                return pybind11::detail::cast_ref<ret_type>(std::move(o), caster);                \
            } else {                                                                              \
                return pybind11::detail::cast_safe<ret_type>(std::move(o));                       \
            }                                                                                     \
            PYBIND11_WARNING_POP                                                                  \
        }                                                                                         \
    } while (false)

/** \rst
    Macro to populate the virtual method in the trampoline class. This macro tries to look up a
    method named 'fn' from the Python side, deals with the :ref:`gil` and necessary argument
    conversions to call this method and return the appropriate type.
    See :ref:`overriding_virtuals` for more information. This macro should be used when the method
    name in C is not the same as the method name in Python. For example with `__str__`.

    .. code-block:: cpp

      std::string toString() override {
        PYBIND11_OVERRIDE_NAME(
            std::string, // Return type (ret_type)
            Animal,      // Parent class (cname)
            "__str__",   // Name of method in Python (name)
            toString,    // Name of function in C++ (fn)
        );
      }
\endrst */
#define PYBIND11_OVERRIDE_NAME(ret_type, cname, name, fn, ...)                                    \
    do {                                                                                          \
        PYBIND11_OVERRIDE_IMPL(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), name, __VA_ARGS__); \
        return cname::fn(__VA_ARGS__);                                                            \
    } while (false)

/** \rst
    Macro for pure virtual functions, this function is identical to
    :c:macro:`PYBIND11_OVERRIDE_NAME`, except that it throws if no override can be found.
\endrst */
#define PYBIND11_OVERRIDE_PURE_NAME(ret_type, cname, name, fn, ...)                               \
    do {                                                                                          \
        PYBIND11_OVERRIDE_IMPL(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), name, __VA_ARGS__); \
        pybind11::pybind11_fail(                                                                  \
            "Tried to call pure virtual function \"" PYBIND11_STRINGIFY(cname) "::" name "\"");   \
    } while (false)

/** \rst
    Macro to populate the virtual method in the trampoline class. This macro tries to look up the
    method from the Python side, deals with the :ref:`gil` and necessary argument conversions to
    call this method and return the appropriate type. This macro should be used if the method name
    in C and in Python are identical.
    See :ref:`overriding_virtuals` for more information.

    .. code-block:: cpp

      class PyAnimal : public Animal {
      public:
          // Inherit the constructors
          using Animal::Animal;

          // Trampoline (need one for each virtual function)
          std::string go(int n_times) override {
              PYBIND11_OVERRIDE_PURE(
                  std::string, // Return type (ret_type)
                  Animal,      // Parent class (cname)
                  go,          // Name of function in C++ (must match Python name) (fn)
                  n_times      // Argument(s) (...)
              );
          }
      };
\endrst */
#define PYBIND11_OVERRIDE(ret_type, cname, fn, ...)                                               \
    PYBIND11_OVERRIDE_NAME(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), #fn, fn, __VA_ARGS__)

/** \rst
    Macro for pure virtual functions, this function is identical to :c:macro:`PYBIND11_OVERRIDE`,
    except that it throws if no override can be found.
\endrst */
#define PYBIND11_OVERRIDE_PURE(ret_type, cname, fn, ...)                                          \
    PYBIND11_OVERRIDE_PURE_NAME(                                                                  \
        PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), #fn, fn, __VA_ARGS__)

// Deprecated versions

PYBIND11_DEPRECATED("get_type_overload has been deprecated")
inline function
get_type_overload(const void *this_ptr, const detail::type_info *this_type, const char *name) {
    return detail::get_type_override(this_ptr, this_type, name);
}

template <class T>
inline function get_overload(const T *this_ptr, const char *name) {
    return get_override(this_ptr, name);
}

#define PYBIND11_OVERLOAD_INT(ret_type, cname, name, ...)                                         \
    PYBIND11_OVERRIDE_IMPL(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), name, __VA_ARGS__)
#define PYBIND11_OVERLOAD_NAME(ret_type, cname, name, fn, ...)                                    \
    PYBIND11_OVERRIDE_NAME(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), name, fn, __VA_ARGS__)
#define PYBIND11_OVERLOAD_PURE_NAME(ret_type, cname, name, fn, ...)                               \
    PYBIND11_OVERRIDE_PURE_NAME(                                                                  \
        PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), name, fn, __VA_ARGS__);
#define PYBIND11_OVERLOAD(ret_type, cname, fn, ...)                                               \
    PYBIND11_OVERRIDE(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), fn, __VA_ARGS__)
#define PYBIND11_OVERLOAD_PURE(ret_type, cname, fn, ...)                                          \
    PYBIND11_OVERRIDE_PURE(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), fn, __VA_ARGS__);

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

#ifndef PYBIND11_PRECOMPILED
#    include "pybind11-inl.h" // IWYU pragma: export
#endif
