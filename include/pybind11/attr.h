/*
    pybind11/attr.h: Infrastructure for processing custom
    type and function attributes

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "cast.h"

NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

/// \addtogroup annotations
/// @{

/// Annotation for methods
struct is_method { handle class_; is_method(const handle &c) : class_(c) { } };

/// Annotation for operators
struct is_operator { };

/// Annotation for parent scope
struct scope { handle value; scope(const handle &s) : value(s) { } };

/// Annotation for documentation
struct doc { const char *value; doc(const char *value) : value(value) { } };

/// Annotation for function names
struct name { const char *value; name(const char *value) : value(value) { } };

/// Annotation indicating that a function is an overload associated with a given "sibling"
struct sibling { handle value; sibling(const handle &value) : value(value.ptr()) { } };

/// Annotation indicating that a class derives from another given type
template <typename T> struct base {
    PYBIND11_DEPRECATED("base<T>() was deprecated in favor of specifying 'T' as a template argument to class_")
    base() { }
};

/// Keep patient alive while nurse lives
template <size_t Nurse, size_t Patient> struct keep_alive { };

/// Annotation indicating that a class is involved in a multiple inheritance relationship
struct multiple_inheritance { };

/// Annotation which enables dynamic attributes, i.e. adds `__dict__` to a class
struct dynamic_attr { };

/// Annotation which enables the buffer protocol for a type
struct buffer_protocol { };

/// Annotation which requests that a special metaclass is created for a type
struct metaclass {
    handle value;

    PYBIND11_DEPRECATED("py::metaclass() is no longer required. It's turned on by default now.")
    metaclass() {}

    /// Override pybind11's default metaclass
    explicit metaclass(handle value) : value(value) { }
};

/// Annotation that marks a class as local to the module:
struct module_local { const bool value; constexpr module_local(bool v = true) : value(v) { } };

/// Annotation to mark enums as an arithmetic type
struct arithmetic { };

/** \rst
    A call policy which places one or more guard variables (``Ts...``) around the function call.

    For example, this definition:

    .. code-block:: cpp

        m.def("foo", foo, py::call_guard<T>());

    is equivalent to the following pseudocode:

    .. code-block:: cpp

        m.def("foo", [](args...) {
            T scope_guard;
            return foo(args...); // forwarded arguments
        });
 \endrst */
template <typename... Ts> struct call_guard;

template <> struct call_guard<> { using type = detail::void_type; };

template <typename T>
struct call_guard<T> {
    static_assert(std::is_default_constructible<T>::value,
                  "The guard type must be default constructible");

    using type = T;
};

template <typename T, typename... Ts>
struct call_guard<T, Ts...> {
    struct type {
        T guard{}; // Compose multiple guard types with left-to-right default-constructor order
        typename call_guard<Ts...>::type next{};
    };
};

/// @} annotations

NAMESPACE_BEGIN(detail)
/* Forward declarations */
enum op_id : int;
enum op_type : int;
struct undefined_t;
template <op_id id, op_type ot, typename L = undefined_t, typename R = undefined_t> struct op_;
template<size_t NumArgs>
void keep_alive_impl(size_t Nurse, size_t Patient, function_call<NumArgs> &call, handle ret);

/// Internal data structure which holds metadata about a keyword argument
struct argument_record {
    const char *name;  ///< Argument name
    const char *descr; ///< Human-readable version of the argument value
    handle value;      ///< Associated Python object
    bool convert : 1;  ///< True if the argument is allowed to convert when loading
    bool none : 1;     ///< True if None is allowed when loading

    argument_record(const char *name, const char *descr, handle value, bool convert, bool none)
        : name(name), descr(descr), value(value), convert(convert), none(none) { }
};

struct invoke_params
{
    const function_record* ptr;
    handle parent;
    value_and_holder* self_value_and_holder;
    size_t n_args_in;
    PyObject* args_in;
    PyObject* kwargs_in;
    bool convert;
};

/// Internal data structure which holds metadata about a bound function (signature, overloads, etc.)
struct function_record {
    function_record()
        : is_constructor(false), is_new_style_constructor(false), is_stateless(false),
          is_operator(false), is_method(false) { }

    /// Function name
    char *name = nullptr; /* why no C++ strings? They generate heavier code.. */

    // User-specified documentation string
    char *doc = nullptr;

    /// Human-readable version of the function signature
    char *signature = nullptr;

    /// List of registered keyword arguments
    std::vector<argument_record> args;

    /// Pointer to lambda function which converts arguments and performs the actual call
    handle(*try_invoke)(const invoke_params&) = nullptr;

    /// Storage for the wrapped function pointer and captured data, if any
    void *data[3] = { };

    /// Pointer to custom destructor for 'data' (if needed)
    void (*free_data) (function_record *ptr) = nullptr;

    /// Python method object
    PyMethodDef *def = nullptr;

    /// Python handle to the parent scope (a class or a module)
    handle scope;

    /// Python handle to the sibling function representing an overload chain
    handle sibling;

    /// Pointer to next overload
    function_record *next = nullptr;

    /// Return value policy associated with this function
    return_value_policy policy = return_value_policy::automatic;

    /// True if name == '__init__'
    bool is_constructor : 1;

    /// True if this is a new-style `__init__` defined in `detail/init.h`
    bool is_new_style_constructor : 1;

    /// True if this is a stateless function pointer
    bool is_stateless : 1;

    /// True if this is an operator (__add__), etc.
    bool is_operator : 1;

    /// True if this is a method
    bool is_method : 1;

    /// Fill in function_call members, return true we can proceed with execution, false is we should continue
    /// with the next candidate
    template<bool HasArgs, bool HasKwargs, size_t NumArgs>
    PYBIND11_NOINLINE bool prepare_function_call(function_call<NumArgs>& call, const invoke_params& params) const
    {
        /* For each overload:
           0. Inject new-style `self` argument
           1. Copy all positional arguments we were given, also checking to make sure that
              named positional arguments weren't *also* specified via kwarg.
           2. If we weren't given enough, try to make up the omitted ones by checking
              whether they were provided by a kwarg matching the `py::arg("name")` name.  If
              so, use it (and remove it from kwargs; if not, see if the function binding
              provided a default that we can use.
           3. Ensure that either all keyword arguments were "consumed", or that the function
              takes a kwargs argument to accept unconsumed kwargs.
           4. Any positional arguments still left get put into a tuple (for args), and any
              leftover kwargs get put into a dict.
         */

        size_t pos_args = NumArgs;    // Number of positional arguments that we need
        if (HasArgs) --pos_args;   // (but don't count py::args
        if (HasKwargs) --pos_args; //  or py::kwargs)

        if (!HasArgs && params.n_args_in > pos_args)
            return false; // Too many arguments for this overload

        if (params.n_args_in < pos_args && args.size() < pos_args)
            return false; // Not enough arguments given, and not enough defaults to fill in the blanks

        size_t args_to_copy = (std::min)(pos_args, params.n_args_in); // Protect std::min with parentheses
        size_t args_copied = 0;

        // 0. Inject new-style `self` argument
        if (is_new_style_constructor) {
            // The `value` may have been preallocated by an old-style `__init__`
            // if it was a preceding candidate for overload resolution.
            if (*params.self_value_and_holder)
                params.self_value_and_holder->type->dealloc(*params.self_value_and_holder);

            call.init_self = PyTuple_GET_ITEM(params.args_in, 0);
            call.args[args_copied] = reinterpret_cast<PyObject*>(params.self_value_and_holder);
            call.args_convert.set(args_copied, false);
            ++args_copied;
        }

        // 1. Copy any position arguments given.
        for (; args_copied < args_to_copy; ++args_copied) {
            const argument_record* arg_rec = args_copied < args.size() ? &args[args_copied] : nullptr;
            if (params.kwargs_in && arg_rec && arg_rec->name && PyDict_GetItemString(params.kwargs_in, arg_rec->name)) {
                return false; // Maybe it was meant for another overload (issue #688)
            }

            handle arg(PyTuple_GET_ITEM(params.args_in, args_copied));

            if (arg_rec && !arg_rec->none && arg.is_none()) {
                return false; // Maybe it was meant for another overload (issue #688)
            }
            call.args[args_copied] = arg;
            call.args_convert.set(args_copied, params.convert && (arg_rec ? arg_rec->convert : true));
        }

        // We'll need to copy this if we steal some kwargs for defaults
        dict kwargs = reinterpret_borrow<dict>(params.kwargs_in);

        // 2. Check kwargs and, failing that, defaults that may help complete the list
        if (args_copied < pos_args) {
            bool copied_kwargs = false;

            for (; args_copied < pos_args; ++args_copied) {
                const auto& arg = args[args_copied];

                handle value;
                if (params.kwargs_in && arg.name)
                    value = PyDict_GetItemString(kwargs.ptr(), arg.name);

                if (value) {
                    // Consume a kwargs value
                    if (!copied_kwargs) {
                        kwargs = reinterpret_steal<dict>(PyDict_Copy(kwargs.ptr()));
                        copied_kwargs = true;
                    }
                    PyDict_DelItemString(kwargs.ptr(), arg.name);
                }
                else if (arg.value) {
                    value = arg.value;
                }

                if (value) {
                    call.args[args_copied] = value;
                    call.args_convert.set(args_copied, params.convert && arg.convert);
                }
                else
                    break;
            }

            if (args_copied < pos_args)
                return false; // Not enough arguments, defaults, or kwargs to fill the positional arguments
        }

        // 3. Check everything was consumed (unless we have a kwargs arg)
        if (!HasKwargs && kwargs && kwargs.size() > 0)
            return false; // Unconsumed kwargs, but no py::kwargs argument to accept them

        // 4a. If we have a py::args argument, create a new tuple with leftovers
        if (HasArgs) {
            tuple extra_args;
            if (args_to_copy == 0) {
                // We didn't copy out any position arguments from the args_in tuple, so we
                // can reuse it directly without copying:
                extra_args = reinterpret_borrow<tuple>(params.args_in);
            }
            else if (args_copied >= params.n_args_in) {
                extra_args = tuple(0);
            }
            else {
                size_t args_size = params.n_args_in - args_copied;
                extra_args = tuple(args_size);
                for (size_t i = 0; i < args_size; ++i) {
                    extra_args[i] = PyTuple_GET_ITEM(params.args_in, args_copied + i);
                }
            }
            call.args[args_copied] = extra_args;
            call.args_convert.set(args_copied, false);
            call.args_ref = std::move(extra_args);
            ++args_copied;
        }

        // 4b. If we have a py::kwargs, pass on any remaining kwargs
        if (HasKwargs) {
            if (!kwargs.ptr())
                kwargs = dict(); // If we didn't get one, send an empty one
            call.args[args_copied] = kwargs;
            call.args_convert.set(args_copied, false);
            call.kwargs_ref = std::move(kwargs);
            ++args_copied;
        }

#if !defined(NDEBUG)
        if (args_copied != NumArgs)
            pybind11_fail("Internal error: function call dispatcher inserted wrong number of arguments!");
#endif
        return true;
    }
};

/// Special data structure which (temporarily) holds metadata about a bound class
struct type_record {
    PYBIND11_NOINLINE type_record()
        : multiple_inheritance(false), dynamic_attr(false), buffer_protocol(false),
          default_holder(true), module_local(false) { }

    /// Handle to the parent scope
    handle scope;

    /// Name of the class
    const char *name = nullptr;

    // Pointer to RTTI type_info data structure
    const std::type_info *type = nullptr;

    /// How large is the underlying C++ type?
    size_t type_size = 0;

    /// What is the alignment of the underlying C++ type?
    size_t type_align = 0;

    /// How large is the type's holder?
    size_t holder_size = 0;

    /// The global operator new can be overridden with a class-specific variant
    void *(*operator_new)(size_t) = nullptr;

    /// Function pointer to class_<..>::init_instance
    void (*init_instance)(instance *, const void *) = nullptr;

    /// Function pointer to class_<..>::dealloc
    void (*dealloc)(detail::value_and_holder &) = nullptr;

    /// List of base classes of the newly created type
    list bases;

    /// Optional docstring
    const char *doc = nullptr;

    /// Custom metaclass (optional)
    handle metaclass;

    /// Multiple inheritance marker
    bool multiple_inheritance : 1;

    /// Does the class manage a __dict__?
    bool dynamic_attr : 1;

    /// Does the class implement the buffer protocol?
    bool buffer_protocol : 1;

    /// Is the default (unique_ptr) holder type used?
    bool default_holder : 1;

    /// Is the class definition local to the module shared object?
    bool module_local : 1;

    PYBIND11_NOINLINE void add_base(const std::type_info &base, void *(*caster)(void *)) {
        auto base_info = detail::get_type_info(base, false);
        if (!base_info) {
            std::string tname(base.name());
            detail::clean_type_id(tname);
            pybind11_fail("generic_type: type \"" + std::string(name) +
                          "\" referenced unknown base type \"" + tname + "\"");
        }

        if (default_holder != base_info->default_holder) {
            std::string tname(base.name());
            detail::clean_type_id(tname);
            pybind11_fail("generic_type: type \"" + std::string(name) + "\" " +
                    (default_holder ? "does not have" : "has") +
                    " a non-default holder type while its base \"" + tname + "\" " +
                    (base_info->default_holder ? "does not" : "does"));
        }

        bases.append((PyObject *) base_info->type);

        if (base_info->type->tp_dictoffset != 0)
            dynamic_attr = true;

        if (caster)
            base_info->implicit_casts.emplace_back(type, caster);
    }
};

/// Tag for a new-style `__init__` defined in `detail/init.h`
struct is_new_style_constructor { };

/**
 * Partial template specializations to process custom attributes provided to
 * cpp_function_ and class_. These are either used to initialize the respective
 * fields in the type_record and function_record data structures or executed at
 * runtime to deal with custom call policies (e.g. keep_alive).
 */
template <typename T, typename SFINAE = void> struct process_attribute;

template <typename T> struct process_attribute_default {
    /// Default implementation: do nothing
    static void init(const T &, function_record *) { }
    static void init(const T &, type_record *) { }
    template<size_t NumArgs> static void precall(function_call<NumArgs> &) { }
    template<size_t NumArgs> static void postcall(function_call<NumArgs> &, handle) { }
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
    static void init(const is_method &s, function_record *r) { r->is_method = true; r->scope = s.class_; }
};

/// Process an attribute which indicates the parent scope of a method
template <> struct process_attribute<scope> : process_attribute_default<scope> {
    static void init(const scope &s, function_record *r) { r->scope = s.value; }
};

/// Process an attribute which indicates that this function is an operator
template <> struct process_attribute<is_operator> : process_attribute_default<is_operator> {
    static void init(const is_operator &, function_record *r) { r->is_operator = true; }
};

template <> struct process_attribute<is_new_style_constructor> : process_attribute_default<is_new_style_constructor> {
    static void init(const is_new_style_constructor &, function_record *r) { r->is_new_style_constructor = true; }
};

/// Process a keyword argument attribute (*without* a default value)
template <> struct process_attribute<arg> : process_attribute_default<arg> {
    static void init(const arg &a, function_record *r) {
        if (r->is_method && r->args.empty())
            r->args.emplace_back("self", nullptr, handle(), true /*convert*/, false /*none not allowed*/);
        r->args.emplace_back(a.name, nullptr, handle(), !a.flag_noconvert, a.flag_none);
    }
};

/// Process a keyword argument attribute (*with* a default value)
template <> struct process_attribute<arg_v> : process_attribute_default<arg_v> {
    static void init(const arg_v &a, function_record *r) {
        if (r->is_method && r->args.empty())
            r->args.emplace_back("self", nullptr /*descr*/, handle() /*parent*/, true /*convert*/, false /*none not allowed*/);

        if (!a.value) {
#if !defined(NDEBUG)
            std::string descr("'");
            if (a.name) descr += std::string(a.name) + ": ";
            descr += a.type + "'";
            if (r->is_method) {
                if (r->name)
                    descr += " in method '" + (std::string) str(r->scope) + "." + (std::string) r->name + "'";
                else
                    descr += " in method of '" + (std::string) str(r->scope) + "'";
            } else if (r->name) {
                descr += " in function '" + (std::string) r->name + "'";
            }
            pybind11_fail("arg(): could not convert default argument "
                          + descr + " into a Python object (type not registered yet?)");
#else
            pybind11_fail("arg(): could not convert default argument "
                          "into a Python object (type not registered yet?). "
                          "Compile in debug mode for more information.");
#endif
        }
        r->args.emplace_back(a.name, a.descr, a.value.inc_ref(), !a.flag_noconvert, a.flag_none);
    }
};

/// Process a parent class attribute.  Single inheritance only (class_ itself already guarantees that)
template <typename T>
struct process_attribute<T, enable_if_t<is_pyobject<T>::value>> : process_attribute_default<handle> {
    static void init(const handle &h, type_record *r) { r->bases.append(h); }
};

/// Process a parent class attribute (deprecated, does not support multiple inheritance)
template <typename T>
struct process_attribute<base<T>> : process_attribute_default<base<T>> {
    static void init(const base<T> &, type_record *r) { r->add_base(typeid(T), nullptr); }
};

/// Process a multiple inheritance attribute
template <>
struct process_attribute<multiple_inheritance> : process_attribute_default<multiple_inheritance> {
    static void init(const multiple_inheritance &, type_record *r) { r->multiple_inheritance = true; }
};

template <>
struct process_attribute<dynamic_attr> : process_attribute_default<dynamic_attr> {
    static void init(const dynamic_attr &, type_record *r) { r->dynamic_attr = true; }
};

template <>
struct process_attribute<buffer_protocol> : process_attribute_default<buffer_protocol> {
    static void init(const buffer_protocol &, type_record *r) { r->buffer_protocol = true; }
};

template <>
struct process_attribute<metaclass> : process_attribute_default<metaclass> {
    static void init(const metaclass &m, type_record *r) { r->metaclass = m.value; }
};

template <>
struct process_attribute<module_local> : process_attribute_default<module_local> {
    static void init(const module_local &l, type_record *r) { r->module_local = l.value; }
};

/// Process an 'arithmetic' attribute for enums (does nothing here)
template <>
struct process_attribute<arithmetic> : process_attribute_default<arithmetic> {};

template <typename... Ts>
struct process_attribute<call_guard<Ts...>> : process_attribute_default<call_guard<Ts...>> { };

/**
 * Process a keep_alive call policy -- invokes keep_alive_impl during the
 * pre-call handler if both Nurse, Patient != 0 and use the post-call handler
 * otherwise
 */
template <size_t Nurse, size_t Patient> struct process_attribute<keep_alive<Nurse, Patient>> : public process_attribute_default<keep_alive<Nurse, Patient>> {
    template <size_t NumArgs, size_t N = Nurse, size_t P = Patient, enable_if_t<N != 0 && P != 0, int> = 0>
    static void precall(function_call<NumArgs> &call) { keep_alive_impl(Nurse, Patient, call, handle()); }
    template <size_t NumArgs, size_t N = Nurse, size_t P = Patient, enable_if_t<N != 0 && P != 0, int> = 0>
    static void postcall(function_call<NumArgs> &, handle) { }
    template <size_t NumArgs, size_t N = Nurse, size_t P = Patient, enable_if_t<N == 0 || P == 0, int> = 0>
    static void precall(function_call<NumArgs> &) { }
    template <size_t NumArgs, size_t N = Nurse, size_t P = Patient, enable_if_t<N == 0 || P == 0, int> = 0>
    static void postcall(function_call<NumArgs> &call, handle ret) { keep_alive_impl(Nurse, Patient, call, ret); }
};

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
    template<size_t NumArgs>
    static void precall(function_call<NumArgs> &call) {
        int unused[] = { 0, (process_attribute<typename std::decay<Args>::type>::precall(call), 0) ... };
        ignore_unused(unused);
    }
    template<size_t NumArgs>
    static void postcall(function_call<NumArgs> &call, handle fn_ret) {
        int unused[] = { 0, (process_attribute<typename std::decay<Args>::type>::postcall(call, fn_ret), 0) ... };
        ignore_unused(unused);
    }
};

template <typename T>
using is_call_guard = is_instantiation<call_guard, T>;

/// Extract the ``type`` from the first `call_guard` in `Extras...` (or `void_type` if none found)
template <typename... Extra>
using extract_guard_t = typename exactly_one_t<is_call_guard, call_guard<>, Extra...>::type;

/// Check the number of named arguments at compile time
template <typename... Extra,
          size_t named = constexpr_sum(std::is_base_of<arg, Extra>::value...),
          size_t self  = constexpr_sum(std::is_same<is_method, Extra>::value...)>
constexpr bool expected_num_args(size_t nargs, bool has_args, bool has_kwargs) {
    return named == 0 || (self + named + has_args + has_kwargs) == nargs;
}

NAMESPACE_END(detail)
NAMESPACE_END(PYBIND11_NAMESPACE)
