/*
    pybind11/attr.h: Infrastructure for processing custom
    type and function attributes

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "cast.h"
#include "options.h"

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
template <size_t NumArgs> void keep_alive_impl(size_t Nurse, size_t Patient, function_call<NumArgs> &call, handle ret);
template <typename... Args> struct process_attributes;

template <typename T>
using is_call_guard = is_instantiation<call_guard, T>;

/// Extract the ``type`` from the first `call_guard` in `Extras...` (or `void_type` if none found)
template <typename... Extra>
using extract_guard_t = typename exactly_one_t<is_call_guard, call_guard<>, Extra...>::type;


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

/// Internal data structure which holds metadata about a bound function (signature, overloads, etc.)
struct function_record {
    function_record()
        : is_constructor(false), is_new_style_constructor(false),
          is_operator(false), is_method(false) { }

    function_record(const function_record&) = delete;
    function_record(function_record&&) = delete;
    
    function_record & operator=(const function_record&) = delete;
    function_record & operator=(function_record&&) = delete;

    virtual ~function_record() {}

    virtual const void* try_get_function_pointer(const std::type_info& function_pointer_type_info) const = 0;
    virtual handle try_invoke(
        handle parent, 
        value_and_holder& self_value_and_holder, 
        size_t n_args_in, 
        PyObject* args_in, 
        PyObject* kwargs_in,
        bool no_convert
    ) const = 0;

    /// Function name
    char *name = nullptr; /* why no C++ strings? They generate heavier code.. */

    // User-specified documentation string
    char *doc = nullptr;

    /// Human-readable version of the function signature
    char *signature = nullptr;

    /// List of registered keyword arguments
    std::vector<argument_record> args;

    /// Python method object
    PyMethodDef* def = nullptr;

    /// Python handle to the parent scope (a class or a module)
    handle scope;

    /// Python handle to the sibling function representing an overload chain
    handle sibling;

    /// Pointer to next overload
    function_record* next = nullptr;

    /// Return value policy associated with this function
    return_value_policy policy = return_value_policy::automatic;

    /// True if name == '__init__'
    bool is_constructor : 1;

    /// True if this is a new-style `__init__` defined in `detail/init.h`
    bool is_new_style_constructor : 1;

    /// True if this is an operator (__add__), etc.
    bool is_operator : 1;

    /// True if this is a method
    bool is_method : 1;

    /// Register a function call with Python (generic non-templated code goes here)
    object initialize_generic(const char* _text, const std::type_info* const* _types, size_t _args) 
    {
        /* Create copies of all referenced C-style strings */
        name = strdup(name ? name : "");
        if (doc) doc = strdup(doc);
        for (auto& a : args) {
            if (a.name)
                a.name = strdup(a.name);
            if (a.descr)
                a.descr = strdup(a.descr);
            else if (a.value)
                a.descr = strdup(a.value.attr("__repr__")().cast<std::string>().c_str());
        }

        is_constructor = !strcmp(name, "__init__") || !strcmp(name, "__setstate__");

#if !defined(NDEBUG) && !defined(PYBIND11_DISABLE_NEW_STYLE_INIT_WARNING)
        if (is_constructor && !is_new_style_constructor) {
            const auto class_name = std::string(((PyTypeObject*)scope.ptr())->tp_name);
            const auto func_name = std::string(name);
            PyErr_WarnEx(
                PyExc_FutureWarning,
                ("pybind11-bound class '" + class_name + "' is using an old-style "
                    "placement-new '" + func_name + "' which has been deprecated. See "
                    "the upgrade guide in pybind11's docs. This message is only visible "
                    "when compiled in debug mode.").c_str(), 0
            );
        }
#endif

        /* Generate a proper function signature */
        std::string _signature;
        size_t type_index = 0, arg_index = 0;
        for (auto* pc = _text; *pc != '\0'; ++pc) {
            const auto c = *pc;

            if (c == '{') {
                // Write arg name for everything except *args and **kwargs.
                if (*(pc + 1) == '*')
                    continue;

                if (arg_index < args.size() && args[arg_index].name) {
                    _signature += args[arg_index].name;
                }
                else if (arg_index == 0 && is_method) {
                    _signature += "self";
                }
                else {
                    _signature += "arg" + std::to_string(arg_index - (is_method ? 1 : 0));
                }
                _signature += ": ";
            }
            else if (c == '}') {
                // Write default value if available.
                if (arg_index < args.size() && args[arg_index].descr) {
                    _signature += " = ";
                    _signature += args[arg_index].descr;
                }
                arg_index++;
            }
            else if (c == '%') {
                const std::type_info* t = _types[type_index++];
                if (!t)
                    pybind11_fail("Internal error while parsing type signature (1)");
                if (auto tinfo = detail::get_type_info(*t)) {
                    handle th((PyObject*)tinfo->type);
                    _signature +=
                        th.attr("__module__").cast<std::string>() + "." +
                        th.attr("__qualname__").cast<std::string>(); // Python 3.3+, but we backport it to earlier versions
                }
                else if (is_new_style_constructor && arg_index == 0) {
                    // A new-style `__init__` takes `self` as `value_and_holder`.
                    // Rewrite it to the proper class type.
                    _signature +=
                        scope.attr("__module__").cast<std::string>() + "." +
                        scope.attr("__qualname__").cast<std::string>();
                }
                else {
                    std::string tname(t->name());
                    detail::clean_type_id(tname);
                    _signature += tname;
                }
            }
            else {
                _signature += c;
            }
        }
        if (arg_index != _args || _types[type_index] != nullptr)
            pybind11_fail("Internal error while parsing type signature (2)");

#if PY_MAJOR_VERSION < 3
        if (strcmp(rec->name, "__next__") == 0) {
            std::free(rec->name);
            rec->name = strdup("next");
        }
        else if (strcmp(rec->name, "__bool__") == 0) {
            std::free(rec->name);
            rec->name = strdup("__nonzero__");
        }
#endif
        signature = strdup(_signature.c_str());
        args.shrink_to_fit();

        if (sibling && PYBIND11_INSTANCE_METHOD_CHECK(sibling.ptr()))
            sibling = PYBIND11_INSTANCE_METHOD_GET_FUNCTION(sibling.ptr());

        function_record* chain = nullptr;
        function_record* chain_start = this;
        if (sibling) {
            if (PyCFunction_Check(sibling.ptr())) {
                auto rec_capsule = reinterpret_borrow<capsule>(PyCFunction_GET_SELF(sibling.ptr()));
                chain = (function_record*)(rec_capsule);
                /* Never append a method to an overload chain of a parent class;
                   instead, hide the parent's overloads in this case */
                if (!chain->scope.is(scope))
                    chain = nullptr;
            }
            // Don't trigger for things like the default __init__, which are wrapper_descriptors that we are intentionally replacing
            else if (!sibling.is_none() && name[0] != '_')
                pybind11_fail("Cannot overload existing non-function object \"" + std::string(name) +
                    "\" with a function of the same name");
        }

        object python_function;
        if (!chain) {
            /* No existing overload was found, create a new function object */
            def = new PyMethodDef();
            std::memset(def, 0, sizeof(PyMethodDef));
            def->ml_name = name;
            def->ml_meth = reinterpret_cast<PyCFunction>(reinterpret_cast<void (*) (void)>(*dispatcher));
            def->ml_flags = METH_VARARGS | METH_KEYWORDS;

            capsule rec_capsule(this, [](void* ptr) {
                destruct(reinterpret_cast<function_record*>(ptr));
            });

            object scope_module;
            if (scope) {
                if (hasattr(scope, "__module__")) {
                    scope_module = scope.attr("__module__");
                }
                else if (hasattr(scope, "__name__")) {
                    scope_module = scope.attr("__name__");
                }
            }

            python_function = reinterpret_steal<object>(PyCFunction_NewEx(def, rec_capsule.ptr(), scope_module.ptr()));
            if (!python_function)
                pybind11_fail("cpp_function::cpp_function(): Could not allocate function object");
        }
        else {
            /* Append at the end of the overload chain */
            python_function = reinterpret_borrow<object>(sibling.ptr());
            chain_start = chain;
            if (chain->is_method != is_method)
                pybind11_fail("overloading a method with both static and instance methods is not supported; "
#if defined(NDEBUG)
                    "compile in debug mode for more details"
#else
                    "error while attempting to bind " + std::string(is_method ? "instance" : "static") + " method " +
                    std::string(pybind11::str(scope.attr("__name__"))) + "." + std::string(name) + signature
#endif
                );
            while (chain->next)
                chain = chain->next;
            chain->next = this;
        }

        std::string signatures;
        int index = 0;
        /* Create a nice pydoc rec including all signatures and
           docstrings of the functions in the overload chain */
        if (chain && options::show_function_signatures()) {
            // First a generic signature
            signatures += name;
            signatures += "(*args, **kwargs)\n";
            signatures += "Overloaded function.\n\n";
        }
        // Then specific overload signatures
        bool first_user_def = true;
        for (auto it = chain_start; it != nullptr; it = it->next) {
            if (options::show_function_signatures()) {
                if (index > 0) signatures += "\n";
                if (chain)
                    signatures += std::to_string(++index) + ". ";
                signatures += name;
                signatures += it->signature;
                signatures += "\n";
            }
            if (it->doc && strlen(it->doc) > 0 && options::show_user_defined_docstrings()) {
                // If we're appending another docstring, and aren't printing function signatures, we
                // need to append a newline first:
                if (!options::show_function_signatures()) {
                    if (first_user_def) first_user_def = false;
                    else signatures += "\n";
                }
                if (options::show_function_signatures()) signatures += "\n";
                signatures += it->doc;
                if (options::show_function_signatures()) signatures += "\n";
            }
        }

        /* Install docstring */
        PyCFunctionObject* func = reinterpret_cast<PyCFunctionObject*>(python_function.ptr());
        if (func->m_ml->ml_doc)
            std::free(const_cast<char*>(func->m_ml->ml_doc));
        func->m_ml->ml_doc = strdup(signatures.c_str());

        if (is_method) {
            python_function = reinterpret_steal<object>(PYBIND11_INSTANCE_METHOD_NEW(python_function.ptr(), scope.ptr()));
            if (!python_function)
                pybind11_fail("cpp_function::cpp_function(): Could not allocate instance method object");
        }

        return python_function;
    }

    /// When a cpp_function is GCed, release any memory allocated by pybind11
    static void destruct(detail::function_record* rec) {
        while (rec) {
            detail::function_record* next = rec->next;
            std::free((char*)rec->name);
            std::free((char*)rec->doc);
            std::free((char*)rec->signature);
            for (auto& arg : rec->args) {
                std::free(const_cast<char*>(arg.name));
                std::free(const_cast<char*>(arg.descr));
                arg.value.dec_ref();
            }
            if (rec->def) {
                std::free(const_cast<char*>(rec->def->ml_doc));
                delete rec->def;
            }
            delete rec;
            rec = next;
        }
    }

    /// Main dispatch logic for calls to functions bound using pybind11
    static PyObject* dispatcher(PyObject* self, PyObject* args_in, PyObject* kwargs_in) {
        using namespace detail;

        const function_record* overloads = reinterpret_cast<function_record*>(PyCapsule_GetPointer(self, nullptr));

        /* Need to know how many arguments + keyword arguments there are to pick the right overload */
        const size_t n_args_in = (size_t)PyTuple_GET_SIZE(args_in);

        handle parent = n_args_in > 0 ? PyTuple_GET_ITEM(args_in, 0) : nullptr;
        handle result = PYBIND11_TRY_NEXT_OVERLOAD;

        value_and_holder self_value_and_holder;
        if (overloads->is_constructor) {
            const auto tinfo = get_type_info((PyTypeObject*)overloads->scope.ptr());
            const auto pi = reinterpret_cast<instance*>(parent.ptr());
            self_value_and_holder = pi->get_value_and_holder(tinfo, false);

            if (!self_value_and_holder.type || !self_value_and_holder.inst) {
                PyErr_SetString(PyExc_TypeError, "__init__(self, ...) called with invalid `self` argument");
                return nullptr;
            }

            // If this value is already registered it must mean __init__ is invoked multiple times;
            // we really can't support that in C++, so just ignore the second __init__.
            if (self_value_and_holder.instance_registered())
                return none().release().ptr();
        }

        try {
            // We do this in two passes: in the first pass, we load arguments with `convert=false`;
            // in the second, we allow conversion (except for arguments with an explicit
            // py::arg().noconvert()).  This lets us prefer calls without conversion, with
            // conversion as a fallback.

            // However, if there are no overloads, we can just skip the no-convert pass entirely
            const bool overloaded = overloads->next != nullptr;

            if (overloaded)
            {
                for (const function_record* it = overloads; it != nullptr && result.ptr() == PYBIND11_TRY_NEXT_OVERLOAD; it = it->next) {
                    result = it->try_invoke(parent, self_value_and_holder, n_args_in, args_in, kwargs_in, true);
                }
            }

            for (const function_record* it = overloads; it != nullptr && result.ptr() == PYBIND11_TRY_NEXT_OVERLOAD; it = it->next) {
                result = it->try_invoke(parent, self_value_and_holder, n_args_in, args_in, kwargs_in, false);
            }
        }
        catch (error_already_set & e) {
            e.restore();
            return nullptr;
#if defined(__GNUG__) && !defined(__clang__)
        }
        catch (abi::__forced_unwind&) {
            throw;
#endif
        }
        catch (...) {
            /* When an exception is caught, give each registered exception
               translator a chance to translate it to a Python exception
               in reverse order of registration.

               A translator may choose to do one of the following:

                - catch the exception and call PyErr_SetString or PyErr_SetObject
                  to set a standard (or custom) Python exception, or
                - do nothing and let the exception fall through to the next translator, or
                - delegate translation to the next translator by throwing a new type of exception. */

            auto last_exception = std::current_exception();
            auto& registered_exception_translators = get_internals().registered_exception_translators;
            for (auto& translator : registered_exception_translators) {
                try {
                    translator(last_exception);
                }
                catch (...) {
                    last_exception = std::current_exception();
                    continue;
                }
                return nullptr;
            }
            PyErr_SetString(PyExc_SystemError, "Exception escaped from default exception translator!");
            return nullptr;
        }

        auto append_note_if_missing_header_is_suspected = [](std::string& msg) {
            if (msg.find("std::") != std::string::npos) {
                msg += "\n\n"
                    "Did you forget to `#include <pybind11/stl.h>`? Or <pybind11/complex.h>,\n"
                    "<pybind11/functional.h>, <pybind11/chrono.h>, etc. Some automatic\n"
                    "conversions are optional and require extra headers to be included\n"
                    "when compiling your pybind11 module.";
            }
        };

        if (result.ptr() == PYBIND11_TRY_NEXT_OVERLOAD) {
            if (overloads->is_operator)
                return handle(Py_NotImplemented).inc_ref().ptr();

            std::string msg = std::string(overloads->name) + "(): incompatible " +
                std::string(overloads->is_constructor ? "constructor" : "function") +
                " arguments. The following argument types are supported:\n";

            int ctr = 0;
            for (const function_record* it2 = overloads; it2 != nullptr; it2 = it2->next) {
                msg += "    " + std::to_string(++ctr) + ". ";

                bool wrote_sig = false;
                if (overloads->is_constructor) {
                    // For a constructor, rewrite `(self: Object, arg0, ...) -> NoneType` as `Object(arg0, ...)`
                    std::string sig = it2->signature;
                    size_t start = sig.find('(') + 7; // skip "(self: "
                    if (start < sig.size()) {
                        // End at the , for the next argument
                        size_t end = sig.find(", "), next = end + 2;
                        size_t ret = sig.rfind(" -> ");
                        // Or the ), if there is no comma:
                        if (end >= sig.size()) next = end = sig.find(')');
                        if (start < end && next < sig.size()) {
                            msg.append(sig, start, end - start);
                            msg += '(';
                            msg.append(sig, next, ret - next);
                            wrote_sig = true;
                        }
                    }
                }
                if (!wrote_sig) msg += it2->signature;

                msg += "\n";
            }
            msg += "\nInvoked with: ";
            auto args_ = reinterpret_borrow<tuple>(args_in);
            bool some_args = false;
            for (size_t ti = overloads->is_constructor ? 1 : 0; ti < args_.size(); ++ti) {
                if (!some_args) some_args = true;
                else msg += ", ";
                msg += pybind11::repr(args_[ti]);
            }
            if (kwargs_in) {
                auto kwargs = reinterpret_borrow<dict>(kwargs_in);
                if (kwargs.size() > 0) {
                    if (some_args) msg += "; ";
                    msg += "kwargs: ";
                    bool first = true;
                    for (auto kwarg : kwargs) {
                        if (first) first = false;
                        else msg += ", ";
                        msg += pybind11::str("{}={!r}").format(kwarg.first, kwarg.second);
                    }
                }
            }

            append_note_if_missing_header_is_suspected(msg);
            PyErr_SetString(PyExc_TypeError, msg.c_str());
            return nullptr;
        }
        else if (!result) {
            std::string msg = "Unable to convert function return value to a "
                "Python type! The signature was\n\t";
            msg += overloads->signature;
            append_note_if_missing_header_is_suspected(msg);
            PyErr_SetString(PyExc_TypeError, msg.c_str());
            return nullptr;
        }
        else {
            if (overloads->is_constructor && !self_value_and_holder.holder_constructed()) {
                auto* pi = reinterpret_cast<instance*>(parent.ptr());
                self_value_and_holder.type->init_instance(pi, nullptr);
            }
            return result.ptr();
        }
    }

};

template<typename Func, typename FunctionType, typename Return, typename CastIn, typename CastOut, typename... Extra>
struct function_record_impl : function_record
{
    static constexpr size_t nargs = CastIn::num_args;
    static constexpr bool has_args = CastIn::has_args;
    static constexpr bool has_kwargs = CastIn::has_kwargs;

    typename std::remove_reference<Func>::type m_func;

    /// Special internal constructor for functors, lambda functions, etc.
    function_record_impl(Func&& f, const Extra&... extra) 
        : m_func(std::forward<Func>(f))
    {
        /* Process any user-provided function attributes */
        process_attributes<Extra...>::init(extra..., this);
    }
    
    template<typename F>
    static typename std::enable_if<std::is_convertible<F, FunctionType>::value, const FunctionType>::type
    try_extract_function_pointer(F& func)
    {
        return static_cast<FunctionType>(func);
    }

    template<typename F>
    static typename std::enable_if<!std::is_convertible<F, FunctionType>::value, const FunctionType>::type
        try_extract_function_pointer(F& func)
    {
        return nullptr;
    }

    virtual const void* try_get_function_pointer(const std::type_info& function_pointer_type_info) const override
    {
        if (same_type(typeid(FunctionType), function_pointer_type_info))
            return reinterpret_cast<const void*>(try_extract_function_pointer(m_func));
        else
            return nullptr;
    }

    virtual handle try_invoke(
        handle parent, 
        value_and_holder& self_value_and_holder, 
        size_t n_args_in, 
        PyObject* args_in,
        PyObject* kwargs_in,
        bool no_convert) const override
    {
        /* For each overload:
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
           5. Pack everything into a vector; if we have py::args or py::kwargs, they are an
              extra tuple or dict at the end of the positional arguments.
           6. Call the function call dispatcher (function_record::impl)

           If one of these fail, move on to the next overload and keep trying until we get a
           result other than PYBIND11_TRY_NEXT_OVERLOAD.
         */

        size_t pos_args = nargs;    // Number of positional arguments that we need
        if (has_args) --pos_args;   // (but don't count py::args
        if (has_kwargs) --pos_args; //  or py::kwargs)

        if (!has_args && n_args_in > pos_args)
            return PYBIND11_TRY_NEXT_OVERLOAD; // Too many arguments for this overload

        if (n_args_in < pos_args && args.size() < pos_args)
            return PYBIND11_TRY_NEXT_OVERLOAD; // Not enough arguments given, and not enough defaults to fill in the blanks

        function_call<CastIn::num_args> call(parent);

        size_t args_to_copy = (std::min)(pos_args, n_args_in); // Protect std::min with parentheses
        size_t args_copied = 0;

        // 0. Inject new-style `self` argument
        if (is_new_style_constructor) {
            // The `value` may have been preallocated by an old-style `__init__`
            // if it was a preceding candidate for overload resolution.
            if (self_value_and_holder)
                self_value_and_holder.type->dealloc(self_value_and_holder);

            call.init_self = PyTuple_GET_ITEM(args_in, 0);
            call.args[args_copied] = reinterpret_cast<PyObject*>(&self_value_and_holder);
            call.args_convert.set(args_copied, false);
            ++args_copied;
        }

        // 1. Copy any position arguments given.
        bool bad_arg = false;
        for (; args_copied < args_to_copy; ++args_copied) {
            const argument_record* arg_rec = args_copied < args.size() ? &args[args_copied] : nullptr;
            if (kwargs_in && arg_rec && arg_rec->name && PyDict_GetItemString(kwargs_in, arg_rec->name)) {
                bad_arg = true;
                break;
            }

            handle arg(PyTuple_GET_ITEM(args_in, args_copied));
            if (arg_rec && !arg_rec->none && arg.is_none()) {
                bad_arg = true;
                break;
            }
            call.args[args_copied] = arg;
            call.args_convert.set(args_copied, !no_convert && arg_rec && arg_rec->convert);
        }
        if (bad_arg)
            return PYBIND11_TRY_NEXT_OVERLOAD; // Maybe it was meant for another overload (issue #688)

        // We'll need to copy this if we steal some kwargs for defaults
        dict kwargs = reinterpret_borrow<dict>(kwargs_in);

        // 2. Check kwargs and, failing that, defaults that may help complete the list
        if (args_copied < pos_args) {
            bool copied_kwargs = false;

            for (; args_copied < pos_args; ++args_copied) {
                const auto& arg = args[args_copied];

                handle value;
                if (kwargs_in && arg.name)
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
                    call.args_convert.set(args_copied, !no_convert && arg.convert);
                }
                else
                    break;
            }

            if (args_copied < pos_args)
                return PYBIND11_TRY_NEXT_OVERLOAD; // Not enough arguments, defaults, or kwargs to fill the positional arguments
        }

        // 3. Check everything was consumed (unless we have a kwargs arg)
        if (kwargs && kwargs.size() > 0 && !has_kwargs)
            return PYBIND11_TRY_NEXT_OVERLOAD; // Unconsumed kwargs, but no py::kwargs argument to accept them

        // 4a. If we have a py::args argument, create a new tuple with leftovers
        if (has_args) {
            tuple extra_args;
            if (args_to_copy == 0) {
                // We didn't copy out any position arguments from the args_in tuple, so we
                // can reuse it directly without copying:
                extra_args = reinterpret_borrow<tuple>(args_in);
            }
            else if (args_copied >= n_args_in) {
                extra_args = tuple(0);
            }
            else {
                size_t args_size = n_args_in - args_copied;
                extra_args = tuple(args_size);
                for (size_t i = 0; i < args_size; ++i) {
                    extra_args[i] = PyTuple_GET_ITEM(args_in, args_copied + i);
                }
            }
            call.args[args_copied] = extra_args;
            call.args_convert.set(args_copied, false);
            call.args_ref = std::move(extra_args);
            ++args_copied;
        }

        // 4b. If we have a py::kwargs, pass on any remaining kwargs
        if (has_kwargs) {
            if (!kwargs.ptr())
                kwargs = dict(); // If we didn't get one, send an empty one
            call.args[args_copied] = kwargs;
            call.args_convert.set(args_copied, false);
            call.kwargs_ref = std::move(kwargs);
            ++args_copied;
        }

        // 5. Put everything in a vector.  Not technically step 5, we've been building it
        // in `call.args` all along.
#if !defined(NDEBUG)
        if (args_copied != nargs)
            pybind11_fail("Internal error: function call dispatcher inserted wrong number of arguments!");
#endif

        // 6. Call the function.
        try {
            loader_life_support guard{};
            /* Dispatch code which converts function arguments and performs the actual function call */
            CastIn args_converter;

            /* Try to cast the function arguments into the C++ domain */
            if (!args_converter.load_args(call))
                return PYBIND11_TRY_NEXT_OVERLOAD;

            /* Invoke call policy pre-call hook */
            process_attributes<Extra...>::precall(call);

            /* Override policy for rvalues -- usually to enforce rvp::move on an rvalue */
            return_value_policy _policy = return_value_policy_override<Return>::policy(policy);

            /* Function scope guard -- defaults to the compile-to-nothing `void_type` */
            using Guard = extract_guard_t<Extra...>;

            /* Perform the function call */
            handle result = CastOut::cast(
                std::move(args_converter).template call<Return, Guard>(m_func), _policy, call.parent);

            /* Invoke call policy post-call hook */
            process_attributes<Extra...>::postcall(call, result);

            return result;
        }
        catch (reference_cast_error&) {
            return PYBIND11_TRY_NEXT_OVERLOAD;
        }
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

/// Check the number of named arguments at compile time
template <typename... Extra,
          size_t named = constexpr_sum(std::is_base_of<arg, Extra>::value...),
          size_t self  = constexpr_sum(std::is_same<is_method, Extra>::value...)>
constexpr bool expected_num_args(size_t nargs, bool has_args, bool has_kwargs) {
    return named == 0 || (self + named + has_args + has_kwargs) == nargs;
}

NAMESPACE_END(detail)
NAMESPACE_END(PYBIND11_NAMESPACE)
