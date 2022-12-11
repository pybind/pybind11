/*
    pybind11/attr.h: Infrastructure for processing custom
    type and function attributes

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "detail/common.h"
#include "cast.h"

#include <functional>
#include <iostream>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

/// \addtogroup annotations
/// @{

/// Annotation for methods
struct is_method {
    handle class_;
    explicit is_method(const handle &c) : class_(c) {}
};

/// Annotation for operators
struct is_operator {};

/// Annotation for classes that cannot be subclassed
struct is_final {};

/// Annotation for parent scope
struct scope {
    handle value;
    explicit scope(const handle &s) : value(s) {}
};

/// Annotation for documentation
struct doc {
    const char *value;
    explicit doc(const char *value) : value(value) {}
};

/// Annotation for function names
struct name {
    const char *value;
    explicit name(const char *value) : value(value) {}
};

/// Annotation indicating that a function is an overload associated with a given "sibling"
struct sibling {
    handle value;
    explicit sibling(const handle &value) : value(value.ptr()) {}
};

/// Annotation indicating that a class derives from another given type
template <typename T>
struct base {

    PYBIND11_DEPRECATED(
        "base<T>() was deprecated in favor of specifying 'T' as a template argument to class_")
    base() = default;
};

struct has_no_temporary_casts {};

/// Keep patient alive while nurse lives
template <size_t Nurse, size_t Patient>
struct keep_alive {};

/// Annotation indicating that a class is involved in a multiple inheritance relationship
struct multiple_inheritance {};

/// Annotation which enables dynamic attributes, i.e. adds `__dict__` to a class
struct dynamic_attr {};

/// Annotation which enables the buffer protocol for a type
struct buffer_protocol {};

/// Annotation which requests that a special metaclass is created for a type
struct metaclass {
    handle value;

    PYBIND11_DEPRECATED("py::metaclass() is no longer required. It's turned on by default now.")
    metaclass() = default;

    /// Override pybind11's default metaclass
    explicit metaclass(handle value) : value(value) {}
};

/// Specifies a custom callback with signature `void (PyHeapTypeObject*)` that
/// may be used to customize the Python type.
///
/// The callback is invoked immediately before `PyType_Ready`.
///
/// Note: This is an advanced interface, and uses of it may require changes to
/// work with later versions of pybind11.  You may wish to consult the
/// implementation of `make_new_python_type` in `detail/classes.h` to understand
/// the context in which the callback will be run.
struct custom_type_setup {
    using callback = std::function<void(PyHeapTypeObject *heap_type)>;

    explicit custom_type_setup(callback value) : value(std::move(value)) {}

    callback value;
};

/// Annotation that marks a class as local to the module:
struct module_local {
    const bool value;
    constexpr explicit module_local(bool v = true) : value(v) {}
};

/// Annotation to mark enums as an arithmetic type
struct arithmetic {};

/// Mark a function for addition at the beginning of the existing overload chain instead of the end
struct prepend {};

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
template <typename... Ts>
struct call_guard;

template <>
struct call_guard<> {
    using type = detail::void_type;
};

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

PYBIND11_NAMESPACE_BEGIN(detail)

template <typename T>
using is_sibling = std::is_same<intrinsic_t<T>, sibling>;

/* Forward declarations */
enum op_id : int;
enum op_type : int;
struct undefined_t;
template <op_id id, op_type ot, typename L = undefined_t, typename R = undefined_t>
struct op_;

/// Special data structure which (temporarily) holds metadata about a bound class
struct type_record {
    PYBIND11_NOINLINE type_record()
        : multiple_inheritance(false), dynamic_attr(false), buffer_protocol(false),
          default_holder(true), module_local(false), is_final(false) {}

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

    /// Custom type setup.
    custom_type_setup::callback custom_type_setup_callback;

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

    /// Is the class inheritable from python classes?
    bool is_final : 1;

    PYBIND11_NOINLINE void add_base(const std::type_info &base, void *(*caster)(void *) ) {
        auto *base_info = detail::get_type_info(base, false);
        if (!base_info) {
            std::string tname(base.name());
            detail::clean_type_id(tname);
            pybind11_fail("generic_type: type \"" + std::string(name)
                          + "\" referenced unknown base type \"" + tname + "\"");
        }

        if (default_holder != base_info->default_holder) {
            std::string tname(base.name());
            detail::clean_type_id(tname);
            pybind11_fail("generic_type: type \"" + std::string(name) + "\" "
                          + (default_holder ? "does not have" : "has")
                          + " a non-default holder type while its base \"" + tname + "\" "
                          + (base_info->default_holder ? "does not" : "does"));
        }

        bases.append((PyObject *) base_info->type);

#if PY_VERSION_HEX < 0x030B0000
        dynamic_attr |= base_info->type->tp_dictoffset != 0;
#else
        dynamic_attr |= (base_info->type->tp_flags & Py_TPFLAGS_MANAGED_DICT) != 0;
#endif

        if (caster) {
            base_info->implicit_casts.emplace_back(type, caster);
        }
    }
};

/// Tag for a new-style `__init__` defined in `detail/init.h`
struct is_new_style_constructor {};

/**
 * Partial template specializations to process custom attributes provided to
 * cpp_function_ and class_. These are either used to initialize the respective
 * fields in the type_record and function_record data structures or executed at
 * runtime to deal with custom call policies (e.g. keep_alive).
 */
template <typename T, typename SFINAE = void>
struct process_attribute;

template <typename T>
struct process_attribute_default {
    /// Default implementation: do nothing
    template <typename RecordType>
    static void init(const T &, RecordType &) {}

    template <typename CallArgs>
    static void precall(CallArgs &, handle) {}

    template <typename CallArgs>
    static void postcall(CallArgs &, handle, handle) {}
};

/// Process an attribute specifying the function's name
template <>
struct process_attribute<name> : process_attribute_default<name> {
    template <typename RecordType>
    static void init(const name &n, RecordType &r) {
        r.name = n.value;
    }
};

/// Process an attribute specifying the function's docstring
template <>
struct process_attribute<doc> : process_attribute_default<doc> {
    template <typename RecordType>
    static void init(const doc &n, RecordType &r) {
        r.doc = n.value;
    }
};

/// Process an attribute specifying the function's docstring (provided as a C-style string)
template <>
struct process_attribute<const char *> : process_attribute_default<const char *> {
    template <typename RecordType>
    static void init(const char *d, RecordType &r) {
        r.doc = d;
    }
};
template <>
struct process_attribute<char *> : process_attribute<const char *> {};

/// Process an attribute indicating the function's return value policy
template <>
struct process_attribute<return_value_policy> : process_attribute_default<return_value_policy> {
    template <typename RecordType>
    static void init(const return_value_policy &p, RecordType &r) {
        r.policy = p;
    }
};

/// Process an attribute which indicates that this is an overloaded function associated with a
/// given sibling
template <>
struct process_attribute<sibling> : process_attribute_default<sibling> {};

/// Process an attribute which indicates that this function is a method
template <>
struct process_attribute<is_method> : process_attribute_default<is_method> {
    template <typename RecordType>
    static void init(const is_method &s, RecordType &r) {
        if (RecordType::nargs > 0) {
            if (r.arginfo_index != 0) {
                pybind11_fail("is_method must come before any arguments when mapping");
            }
            r.argument_info[0].name = "self";
            r.argument_info[0].convert = false;
            r.arginfo_index++;
            r.current_scope = s.class_;
            r.argument_index["self"] = -1;
        }
    }
};

/// Process an attribute which indicates the parent scope of a method
template <>
struct process_attribute<scope> : process_attribute_default<scope> {
    template <typename RecordType>
    static void init(const scope &s, RecordType &r) {
        r.current_scope = s.value;
    }
};

/// Process an attribute which indicates that this function is an operator
template <>
struct process_attribute<is_operator> : process_attribute_default<is_operator> {};

template <>
struct process_attribute<has_no_temporary_casts>
    : process_attribute_default<has_no_temporary_casts> {};

template <>
struct process_attribute<is_new_style_constructor>
    : process_attribute_default<is_new_style_constructor> {};

/// Process a keyword argument attribute (*without* a default value)
template <>
struct process_attribute<arg> : process_attribute_default<arg> {
    template <typename RecordType>
    static void init(const arg &a, RecordType &r) {
        if (r.arginfo_index >= RecordType::nargs_pos && (!a.name || a.name[0] == '\0')) {
            pybind11_fail(
                "arg(): cannot specify an unnamed argument after a kw_only() annotation or "
                "args() argument");
        }

        if (r.arginfo_index == RecordType::nargs_pos && RecordType::has_args) {
            r.arginfo_index++;
        }

        if (a.name != nullptr) {
            r.argument_info[r.arginfo_index].name = a.name;
            PYBIND11_WARNING_PUSH
            PYBIND11_WARNING_DISABLE_INTEL(186) // "Pointless" comparison with zero
            if (r.arginfo_index < RecordType::nargs_pos_only) {
                r.argument_index[a.name] = -1;
            } else {
                r.argument_index[a.name] = static_cast<ssize_t>(r.arginfo_index);
            }
            PYBIND11_WARNING_PUSH
        }

        r.argument_info[r.arginfo_index].convert = !a.flag_noconvert;
        r.argument_info[r.arginfo_index].none = a.flag_none;

        r.arginfo_index++;
    }
};

/// Process a keyword argument attribute (*with* a default value)
template <>
struct process_attribute<arg_v> : process_attribute_default<arg_v> {
    template <typename RecordType>
    static void init(const arg_v &a, RecordType &r) {
        process_attribute<arg>::init(a, r);
        if (!a.value) {
#if defined(PYBIND11_DETAILED_ERROR_MESSAGES)
            std::string descr("'");
            if (a.name) {
                descr += std::string(a.name) + ": ";
            }
            descr += a.type + "'";
            if (RecordType::has_self) {
                if (!r.name.empty()) {
                    descr += " in method '" + (std::string) str(r.current_scope) + "."
                             + (std::string) r.name + "'";
                } else {
                    descr += " in method of '" + (std::string) str(r.current_scope) + "'";
                }
            } else if (!r.name.empty()) {
                descr += " in function '" + (std::string) r.name + "'";
            }
            pybind11_fail("arg(): could not convert default argument " + descr
                          + " into a Python object (type not registered yet?)");
#else
            pybind11_fail("arg(): could not convert default argument "
                          "into a Python object (type not registered yet?). "
                          "#define PYBIND11_DETAILED_ERROR_MESSAGES or compile in debug mode for "
                          "more information.");
#endif
        }
        if (a.descr != nullptr) {
            r.argument_info[r.arginfo_index - 1].desc = a.descr;
        } else {
            r.argument_info[r.arginfo_index - 1].desc = repr(a.value).cast<std::string>();
        }
        r.argument_info[r.arginfo_index - 1].value = reinterpret_borrow<object>(a.value);
    }
};

/// Process a keyword-only-arguments-follow pseudo argument
template <>
struct process_attribute<kw_only> : process_attribute_default<kw_only> {
    template <typename RecordType>
    static void init(const kw_only &, RecordType &r) {
        if (RecordType::nargs_pos != r.arginfo_index) {
            throw std::runtime_error("Should never happen as defined at compile time "
                                     + std::to_string(RecordType::nargs_pos) + " "
                                     + std::to_string(r.arginfo_index) + " " + r.name + " "
                                     + std::to_string(RecordType::has_kw_only_args) + " "
                                     + std::to_string(RecordType::kw_only_pos) + " "
                                     + std::to_string(RecordType::has_args));

            // static constexpr size_t nargs_pos = has_kw_only_args ? kw_only_pos : (has_args ?
            // args_pos :
            //  (has_kwargs ? nargs - 1 : nargs)
        }
    }
};

/// Process a positional-only-argument maker
template <>
struct process_attribute<pos_only> : process_attribute_default<pos_only> {
    template <typename RecordType>
    static void init(const pos_only &, RecordType &r) {
        if (RecordType::nargs_pos_only != r.arginfo_index) {
            throw std::runtime_error("Should never happen as defined at compile time pos only "
                                     + std::to_string(r.arginfo_index) + " "
                                     + std::to_string(RecordType::nargs_pos_only));
        }
    }
};

/// Process a parent class attribute.  Single inheritance only (class_ itself already
/// guarantees that)
template <typename T>
struct process_attribute<T, enable_if_t<is_pyobject<T>::value>>
    : process_attribute_default<handle> {
    template <typename RecordType>
    static void init(const handle &h, RecordType &r) {
        r.bases.append(h);
    }
};

/// Process a parent class attribute (deprecated, does not support multiple inheritance)
template <typename T>
struct process_attribute<base<T>> : process_attribute_default<base<T>> {
    template <typename RecordType>
    static void init(const base<T> &, RecordType &r) {
        r.add_base(typeid(T), nullptr);
    }
};

/// Process a multiple inheritance attribute
template <>
struct process_attribute<multiple_inheritance> : process_attribute_default<multiple_inheritance> {
    template <typename RecordType>
    static void init(const multiple_inheritance &, RecordType &r) {
        r.multiple_inheritance = true;
    }
};

template <>
struct process_attribute<dynamic_attr> : process_attribute_default<dynamic_attr> {
    template <typename RecordType>
    static void init(const dynamic_attr &, RecordType &r) {
        r.dynamic_attr = true;
    }
};

template <>
struct process_attribute<custom_type_setup> {
    template <typename RecordType>
    static void init(const custom_type_setup &value, RecordType &r) {
        r.custom_type_setup_callback = value.value;
    }
};

template <>
struct process_attribute<is_final> : process_attribute_default<is_final> {
    template <typename RecordType>
    static void init(const is_final &, RecordType &r) {
        r.is_final = true;
    }
};

template <>
struct process_attribute<buffer_protocol> : process_attribute_default<buffer_protocol> {
    template <typename RecordType>
    static void init(const buffer_protocol &, RecordType &r) {
        r.buffer_protocol = true;
    }
};

template <>
struct process_attribute<metaclass> : process_attribute_default<metaclass> {
    template <typename RecordType>
    static void init(const metaclass &m, RecordType &r) {
        r.metaclass = m.value;
    }
};

template <>
struct process_attribute<module_local> : process_attribute_default<module_local> {
    template <typename RecordType>
    static void init(const module_local &l, RecordType &r) {
        r.module_local = l.value;
    }
};

/// Process a 'prepend' attribute, putting this at the beginning of the overload chain
template <>
struct process_attribute<prepend> : process_attribute_default<prepend> {};

/// Process an 'arithmetic' attribute for enums (does nothing here)
template <>
struct process_attribute<arithmetic> : process_attribute_default<arithmetic> {};

template <typename... Ts>
struct process_attribute<call_guard<Ts...>> : process_attribute_default<call_guard<Ts...>> {};

template <size_t Nurse, size_t Patient, typename CallArgs>
void keep_alive_impl_for_call(const CallArgs &call_args, handle parent, handle ret) {
    static_assert(Nurse <= std::tuple_size<CallArgs>::value,
                  "Nurse must be within the range of call arguments");
    static_assert(Patient <= std::tuple_size<CallArgs>::value,
                  "Patient must be within the range of call arguments");
    auto get_arg = [&](size_t n) {
        if (n == 0) {
            return ret;
        }
        if (n == 1) {
            return parent;
        }
        if (n <= call_args.size()) {
            return call_args[n - 1];
        }
        pybind11_fail("This should never happen, internal pybind11 error");
    };

    keep_alive_impl(get_arg(Nurse), get_arg(Patient));
}

/**
 * Process a keep_alive call policy -- invokes keep_alive_impl during the
 * pre-call handler if both Nurse, Patient != 0 and use the post-call handler
 * otherwise
 */
template <size_t Nurse, size_t Patient>
struct process_attribute<keep_alive<Nurse, Patient>>
    : public process_attribute_default<keep_alive<Nurse, Patient>> {

    template <typename CallArgs,
              size_t N = Nurse,
              size_t P = Patient,
              enable_if_t<N != 0 && P != 0, int> = 0>
    static void precall(CallArgs &call_args, handle parent) {
        keep_alive_impl_for_call<N, P, CallArgs>(call_args, parent, handle());
    }
    template <typename CallArgs,
              size_t N = Nurse,
              size_t P = Patient,
              enable_if_t<N != 0 && P != 0, int> = 0>
    static void postcall(CallArgs &, handle, handle) {}

    template <typename CallArgs,
              size_t N = Nurse,
              size_t P = Patient,
              enable_if_t<N == 0 || P == 0, int> = 0>
    static void precall(CallArgs &, handle) {}

    template <typename CallArgs,
              size_t N = Nurse,
              size_t P = Patient,
              enable_if_t<N == 0 || P == 0, int> = 0>
    static void postcall(CallArgs &call_args, handle parent, handle ret) {
        keep_alive_impl_for_call<N, P, CallArgs>(call_args, parent, ret);
    }
};

/// Recursively iterate over variadic template arguments
template <typename... Args>
struct process_attributes {
    template <typename RecordType>
    static void init(const Args &...args, RecordType &r) {
        PYBIND11_WORKAROUND_INCORRECT_MSVC_C4100(r);
        PYBIND11_WORKAROUND_INCORRECT_GCC_UNUSED_BUT_SET_PARAMETER(r);
        using expander = int[];
        (void) expander{
            0, ((void) process_attribute<typename std::decay<Args>::type>::init(args, r), 0)...};
    }
    template <typename CallArgs>
    static void precall(CallArgs &call_args, handle parent) {
        PYBIND11_WORKAROUND_INCORRECT_MSVC_C4100(call_args);
        PYBIND11_WORKAROUND_INCORRECT_GCC_UNUSED_BUT_SET_PARAMETER(parent);
        using expander = int[];
        (void) expander{
            0,
            (process_attribute<typename std::decay<Args>::type>::precall(call_args, parent),
             0)...};
    }
    template <typename CallArgs>
    static void postcall(CallArgs &call_args, handle parent, handle fn_ret) {
        PYBIND11_WORKAROUND_INCORRECT_MSVC_C4100(call_args, fn_ret);
        PYBIND11_WORKAROUND_INCORRECT_GCC_UNUSED_BUT_SET_PARAMETER(parent);
        PYBIND11_WORKAROUND_INCORRECT_GCC_UNUSED_BUT_SET_PARAMETER(fn_ret);
        using expander = int[];
        (void) expander{0,
                        (process_attribute<typename std::decay<Args>::type>::postcall(
                             call_args, parent, fn_ret),
                         0)...};
    }
};

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
