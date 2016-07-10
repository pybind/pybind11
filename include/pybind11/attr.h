/*
    pybind11/pybind11.h: Infrastructure for processing custom
    type and function attributes

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "cast.h"

NAMESPACE_BEGIN(pybind11)

template <typename T> struct arg_t;

/// Annotation for keyword arguments
struct arg {
    constexpr explicit arg(const char *name) : name(name) { }

    template <typename T>
    constexpr arg_t<T> operator=(const T &value) const { return {name, value}; }
    template <typename T, size_t N>
    constexpr arg_t<const T *> operator=(T const (&value)[N]) const {
        return operator=((const T *) value);
    }

    const char *name;
};

/// Annotation for keyword arguments with default values
template <typename T> struct arg_t : public arg {
    constexpr arg_t(const char *name, const T &value, const char *descr = nullptr)
        : arg(name), value(value), descr(descr) { }
    T value;
    const char *descr;
};

inline namespace literals {
/// String literal version of arg
constexpr arg operator"" _a(const char *name, size_t) { return arg(name); }
}

/// Annotation for methods
struct is_method { handle class_; is_method(const handle &c) : class_(c) { } };

/// Annotation for parent scope
struct scope { handle value; scope(const handle &s) : value(s) { } };

/// Annotation for documentation
struct doc { const char *value; doc(const char *value) : value(value) { } };

/// Annotation for function names
struct name { const char *value; name(const char *value) : value(value) { } };

/// Annotation indicating that a function is an overload associated with a given "sibling"
struct sibling { handle value; sibling(const handle &value) : value(value.ptr()) { } };

/// Annotation indicating that a class derives from another given type
template <typename T> struct base { };

/// Keep patient alive while nurse lives
template <int Nurse, int Patient> struct keep_alive { };

NAMESPACE_BEGIN(detail)
/* Forward declarations */
enum op_id : int;
enum op_type : int;
struct undefined_t;
template <op_id id, op_type ot, typename L = undefined_t, typename R = undefined_t> struct op_;
template <typename... Args> struct init;
template <typename... Args> struct init_alias;
inline void keep_alive_impl(int Nurse, int Patient, handle args, handle ret);

/// Internal data structure which holds metadata about a keyword argument
struct argument_record {
    const char *name;  ///< Argument name
    const char *descr; ///< Human-readable version of the argument value
    handle value;      ///< Associated Python object

    argument_record(const char *name, const char *descr, handle value)
        : name(name), descr(descr), value(value) { }
};

/// Internal data structure which holds metadata about a bound function (signature, overloads, etc.)
struct function_record {
    /// Function name
    char *name = nullptr; /* why no C++ strings? They generate heavier code.. */

    // User-specified documentation string
    char *doc = nullptr;

    /// Human-readable version of the function signature
    char *signature = nullptr;

    /// List of registered keyword arguments
    std::vector<argument_record> args;

    /// Pointer to lambda function which converts arguments and performs the actual call
    handle (*impl) (function_record *, handle, handle, handle) = nullptr;

    /// Storage for the wrapped function pointer and captured data, if any
    void *data[3] = { };

    /// Pointer to custom destructor for 'data' (if needed)
    void (*free_data) (function_record *ptr) = nullptr;

    /// Return value policy associated with this function
    return_value_policy policy = return_value_policy::automatic;

    /// True if name == '__init__'
    bool is_constructor : 1;

    /// True if this is a stateless function pointer
    bool is_stateless : 1;

    /// True if the function has a '*args' argument
    bool has_args : 1;

    /// True if the function has a '**kwargs' argument
    bool has_kwargs : 1;

    /// Number of arguments
    uint16_t nargs;

    /// Python method object
    PyMethodDef *def = nullptr;

    /// Python handle to the associated class (if this is method)
    handle class_;

    /// Python handle to the parent scope (a class or a module)
    handle scope;

    /// Python handle to the sibling function representing an overload chain
    handle sibling;

    /// Pointer to next overload
    function_record *next = nullptr;
};

/// Special data structure which (temporarily) holds metadata about a bound class
struct type_record {
    /// Handle to the parent scope
    handle scope;

    /// Name of the class
    const char *name = nullptr;

    // Pointer to RTTI type_info data structure
    const std::type_info *type = nullptr;

    /// How large is the underlying C++ type?
    size_t type_size = 0;

    /// How large is pybind11::instance<type>?
    size_t instance_size = 0;

    /// Function pointer to class_<..>::init_holder
    void (*init_holder)(PyObject *, const void *) = nullptr;

    /// Function pointer to class_<..>::dealloc
    void (*dealloc)(PyObject *) = nullptr;

    // Pointer to RTTI type_info data structure of base class
    const std::type_info *base_type = nullptr;

    /// OR: Python handle to base class
    handle base_handle;

    /// Optional docstring
    const char *doc = nullptr;
};

/**
 * Partial template specializations to process custom attributes provided to
 * cpp_function_ and class_. These are either used to initialize the respective
 * fields in the type_record and function_record data structures or executed
 * at runtime to deal with custom call policies (e.g. keep_alive).
 */
template <typename T, typename SFINAE = void> struct process_attribute;

template <typename T> struct process_attribute_default {
    /// Default implementation: do nothing
    static void init(const T &, function_record *) { }
    static void init(const T &, type_record *) { }
    static void precall(handle) { }
    static void postcall(handle, handle) { }
};

/// Process an attribute specifying the function's name
template <> struct process_attribute<name> : process_attribute_default<name> {
    static void init(const name &n, function_record *r) { r->name = const_cast<char *>(n.value); }
};

/// Process an attribute specifying the function's docstring
template <> struct process_attribute<doc> : process_attribute_default<doc> {
    static void init(const doc &n, function_record *r) { r->doc = const_cast<char *>(n.value); }
};

/// Process an attribute specifying the function's docstring (provided as a C-style string)
template <> struct process_attribute<const char *> : process_attribute_default<const char *> {
    static void init(const char *d, function_record *r) { r->doc = const_cast<char *>(d); }
    static void init(const char *d, type_record *r) { r->doc = const_cast<char *>(d); }
};
template <> struct process_attribute<char *> : process_attribute<const char *> { };

/// Process an attribute indicating the function's return value policy
template <> struct process_attribute<return_value_policy> : process_attribute_default<return_value_policy> {
    static void init(const return_value_policy &p, function_record *r) { r->policy = p; }
};

/// Process an attribute which indicates that this is an overloaded function associated with a given sibling
template <> struct process_attribute<sibling> : process_attribute_default<sibling> {
    static void init(const sibling &s, function_record *r) { r->sibling = s.value; }
};

/// Process an attribute which indicates that this function is a method
template <> struct process_attribute<is_method> : process_attribute_default<is_method> {
    static void init(const is_method &s, function_record *r) { r->class_ = s.class_; r->scope = s.class_; }
};

/// Process an attribute which indicates the parent scope of a method
template <> struct process_attribute<scope> : process_attribute_default<scope> {
    static void init(const scope &s, function_record *r) { r->scope = s.value; }
};


/// Process a keyword argument attribute (*without* a default value)
template <> struct process_attribute<arg> : process_attribute_default<arg> {
    static void init(const arg &a, function_record *r) {
        if (r->class_ && r->args.empty())
            r->args.emplace_back("self", nullptr, handle());
        r->args.emplace_back(a.name, nullptr, handle());
    }
};

/// Process a keyword argument attribute (*with* a default value)
template <typename T>
struct process_attribute<arg_t<T>> : process_attribute_default<arg_t<T>> {
    static void init(const arg_t<T> &a, function_record *r) {
        if (r->class_ && r->args.empty())
            r->args.emplace_back("self", nullptr, handle());

        /* Convert keyword value into a Python object */
        object o = object(detail::type_caster<typename detail::intrinsic_type<T>::type>::cast(
                a.value, return_value_policy::automatic, handle()), false);

        if (!o) {
#if !defined(NDEBUG)
            std::string descr(typeid(T).name());
            detail::clean_type_id(descr);
            descr = "'" + std::string(a.name) + ": " + descr + "'";
            if (r->class_) {
                if (r->name)
                    descr += " in method '" + (std::string) r->class_.str() + "." + (std::string) r->name + "'";
                else
                    descr += " in method of '" + (std::string) r->class_.str() + "'";
            } else if (r->name) {
                descr += " in function named '" + (std::string) r->name + "'";
            }
            pybind11_fail("arg(): could not convert default keyword argument "
                          + descr + " into a Python object (type not registered yet?)");
#else
            pybind11_fail("arg(): could not convert default keyword argument "
                          "into a Python object (type not registered yet?). "
                          "Compile in debug mode for more information.");
#endif
        }
        r->args.emplace_back(a.name, a.descr, o.release());
    }
};

/// Process a parent class attribute
template <typename T>
struct process_attribute<T, typename std::enable_if<std::is_base_of<handle, T>::value>::type> : process_attribute_default<handle> {
    static void init(const handle &h, type_record *r) { r->base_handle = h; }
};

/// Process a parent class attribute
template <typename T>
struct process_attribute<base<T>> : process_attribute_default<base<T>> {
    static void init(const base<T> &, type_record *r) { r->base_type = &typeid(T); }
};

/***
 * Process a keep_alive call policy -- invokes keep_alive_impl during the
 * pre-call handler if both Nurse, Patient != 0 and use the post-call handler
 * otherwise
 */
template <int Nurse, int Patient> struct process_attribute<keep_alive<Nurse, Patient>> : public process_attribute_default<keep_alive<Nurse, Patient>> {
    template <int N = Nurse, int P = Patient, typename std::enable_if<N != 0 && P != 0, int>::type = 0>
    static void precall(handle args) { keep_alive_impl(Nurse, Patient, args, handle()); }
    template <int N = Nurse, int P = Patient, typename std::enable_if<N != 0 && P != 0, int>::type = 0>
    static void postcall(handle, handle) { }
    template <int N = Nurse, int P = Patient, typename std::enable_if<N == 0 || P == 0, int>::type = 0>
    static void precall(handle) { }
    template <int N = Nurse, int P = Patient, typename std::enable_if<N == 0 || P == 0, int>::type = 0>
    static void postcall(handle args, handle ret) { keep_alive_impl(Nurse, Patient, args, ret); }
};

/// Ignore that a variable is unused in compiler warnings
inline void ignore_unused(const int *) { }

/// Recursively iterate over variadic template arguments
template <typename... Args> struct process_attributes {
    static void init(const Args&... args, function_record *r) {
        int unused[] = { 0, (process_attribute<typename std::decay<Args>::type>::init(args, r), 0) ... };
        ignore_unused(unused);
    }
    static void init(const Args&... args, type_record *r) {
        int unused[] = { 0, (process_attribute<typename std::decay<Args>::type>::init(args, r), 0) ... };
        ignore_unused(unused);
    }
    static void precall(handle fn_args) {
        int unused[] = { 0, (process_attribute<typename std::decay<Args>::type>::precall(fn_args), 0) ... };
        ignore_unused(unused);
    }
    static void postcall(handle fn_args, handle fn_ret) {
        int unused[] = { 0, (process_attribute<typename std::decay<Args>::type>::postcall(fn_args, fn_ret), 0) ... };
        ignore_unused(unused);
    }
};

/// Compile-time integer sum
constexpr size_t constexpr_sum() { return 0; }
template <typename T, typename... Ts>
constexpr size_t constexpr_sum(T n, Ts... ns) { return n + constexpr_sum(ns...); }

/// Check the number of named arguments at compile time
template <typename... Extra,
          size_t named = constexpr_sum(std::is_base_of<arg, Extra>::value...),
          size_t self  = constexpr_sum(std::is_same<is_method, Extra>::value...)>
constexpr bool expected_num_args(size_t nargs) {
    return named == 0 || (self + named) == nargs;
}

NAMESPACE_END(detail)
NAMESPACE_END(pybind11)
