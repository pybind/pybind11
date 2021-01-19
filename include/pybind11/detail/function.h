/*
    pybind11/detail/function.h: implementation of C++ function type

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"
#include "../attr.h"
#include "../options.h"
#include "../pytypes.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)
/// Generate a string containing the signature of the function
PYBIND11_NOINLINE inline std::string make_func_signature(const detail::function_record *rec,
                                                         const char *text,
                                                         const std::type_info *const *types,
                                                         size_t args) {
    std::string signature;
    size_t type_index = 0, arg_index = 0;
    for (auto *pc = text; *pc != '\0'; ++pc) {
        const auto c = *pc;

        if (c == '{') {
            // Write arg name for everything except *args and **kwargs.
            if (*(pc + 1) == '*')
                continue;
            // Separator for keyword-only arguments, placed before the kw
            // arguments start
            if (rec->nargs_kw_only > 0 && arg_index + rec->nargs_kw_only == args)
                signature += "*, ";
            if (arg_index < rec->args.size() && rec->args[arg_index].name) {
                signature += rec->args[arg_index].name;
            } else if (arg_index == 0 && rec->is_method) {
                signature += "self";
            } else {
                signature += "arg" + std::to_string(arg_index - (rec->is_method ? 1 : 0));
            }
            signature += ": ";
        } else if (c == '}') {
            // Write default value if available.
            if (arg_index < rec->args.size() && rec->args[arg_index].descr) {
                signature += " = ";
                signature += rec->args[arg_index].descr;
            }
            // Separator for positional-only arguments (placed after the
            // argument, rather than before like *
            if (rec->nargs_pos_only > 0 && (arg_index + 1) == rec->nargs_pos_only)
                signature += ", /";
            arg_index++;
        } else if (c == '%') {
            const std::type_info *t = types[type_index++];
            if (!t)
                pybind11_fail("Internal error while parsing type signature (1)");
            if (auto tinfo = detail::get_type_info(*t)) {
                handle th((PyObject *) tinfo->type);
                signature +=
                    th.attr("__module__").cast<std::string>() + "." +
                    th.attr("__qualname__").cast<std::string>(); // Python 3.3+, but we backport it to earlier versions
            } else if (rec->is_new_style_constructor && arg_index == 0) {
                // A new-style `__init__` takes `self` as `value_and_holder`.
                // Rewrite it to the proper class type.
                signature +=
                    rec->scope.attr("__module__").cast<std::string>() + "." +
                    rec->scope.attr("__qualname__").cast<std::string>();
            } else {
                std::string tname(t->name());
                detail::clean_type_id(tname);
                signature += tname;
            }
        } else {
            signature += c;
        }
    }

    if (arg_index != args || types[type_index] != nullptr)
        pybind11_fail("Internal error while parsing type signature (2)");

    return signature;
}

/// Generate docstring from the signature and description of function overloads
PYBIND11_NOINLINE inline std::string make_func_doc(const detail::function_record *rec,
                                                   bool has_chain,
                                                   const detail::function_record *chain_start) {
    std::string doc;
    int index = 0;
    /* Create a nice pydoc rec including all signatures and
       docstrings of the functions in the overload chain */
    if (has_chain && options::show_function_signatures()) {
        // First a generic signature
        doc += rec->name;
        doc += "(*args, **kwargs)\n";
        doc += "Overloaded function.\n\n";
    }
    // Then specific overload signatures
    bool first_user_def = true;
    for (auto it = chain_start; it != nullptr; it = it->next) {
        if (options::show_function_signatures()) {
            if (index > 0) doc += "\n";
            if (has_chain)
                doc += std::to_string(++index) + ". ";
            doc += rec->name;
            doc += it->signature;
            doc += "\n";
        }
        if (it->doc && strlen(it->doc) > 0 && options::show_user_defined_docstrings()) {
            // If we're appending another docstring, and aren't printing function signatures, we
            // need to append a newline first:
            if (!options::show_function_signatures()) {
                if (first_user_def) first_user_def = false;
                else doc += "\n";
            }
            if (options::show_function_signatures()) doc += "\n";
            doc += it->doc;
            if (options::show_function_signatures()) doc += "\n";
        }
    }
    return doc;
}
PYBIND11_NAMESPACE_END(detail)

/// Wraps an arbitrary C++ function/method/lambda function/.. into a callable Python object
class cpp_function : public function {
public:
    cpp_function() = default;
    cpp_function(std::nullptr_t) { }

    /// Construct a cpp_function from a vanilla function pointer
    template <typename Return, typename... Args, typename... Extra>
    cpp_function(Return (*f)(Args...), const Extra&... extra) {
        initialize(f, f, extra...);
    }

    /// Construct a cpp_function from a lambda function (possibly with internal state)
    template <typename Func, typename... Extra,
              typename = detail::enable_if_t<detail::is_lambda<Func>::value>>
    cpp_function(Func &&f, const Extra&... extra) {
        initialize(std::forward<Func>(f),
                   (detail::function_signature_t<Func> *) nullptr, extra...);
    }

    /// Construct a cpp_function from a class method (non-const, no ref-qualifier)
    template <typename Return, typename Class, typename... Arg, typename... Extra>
    cpp_function(Return (Class::*f)(Arg...), const Extra&... extra) {
        initialize([f](Class *c, Arg... args) -> Return { return (c->*f)(std::forward<Arg>(args)...); },
                   (Return (*) (Class *, Arg...)) nullptr, extra...);
    }

    /// Construct a cpp_function from a class method (non-const, lvalue ref-qualifier)
    /// A copy of the overload for non-const functions without explicit ref-qualifier
    /// but with an added `&`.
    template <typename Return, typename Class, typename... Arg, typename... Extra>
    cpp_function(Return (Class::*f)(Arg...)&, const Extra&... extra) {
        initialize([f](Class *c, Arg... args) -> Return { return (c->*f)(args...); },
                   (Return (*) (Class *, Arg...)) nullptr, extra...);
    }

    /// Construct a cpp_function from a class method (const, no ref-qualifier)
    template <typename Return, typename Class, typename... Arg, typename... Extra>
    cpp_function(Return (Class::*f)(Arg...) const, const Extra&... extra) {
        initialize([f](const Class *c, Arg... args) -> Return { return (c->*f)(std::forward<Arg>(args)...); },
                   (Return (*)(const Class *, Arg ...)) nullptr, extra...);
    }

    /// Construct a cpp_function from a class method (const, lvalue ref-qualifier)
    /// A copy of the overload for const functions without explicit ref-qualifier
    /// but with an added `&`.
    template <typename Return, typename Class, typename... Arg, typename... Extra>
    cpp_function(Return (Class::*f)(Arg...) const&, const Extra&... extra) {
        initialize([f](const Class *c, Arg... args) -> Return { return (c->*f)(args...); },
                   (Return (*)(const Class *, Arg ...)) nullptr, extra...);
    }

    /// Return the function name
    object name() const { return attr("__name__"); }

protected:
    struct InitializingFunctionRecordDeleter {
        // `destruct(function_record, false)`: `initialize_generic` copies strings and
        // takes care of cleaning up in case of exceptions. So pass `false` to `free_strings`.
        void operator()(detail::function_record * rec) { destruct(rec, false); }
    };
    using unique_function_record = std::unique_ptr<detail::function_record, InitializingFunctionRecordDeleter>;

    /// Space optimization: don't inline this frequently instantiated fragment
    PYBIND11_NOINLINE unique_function_record make_function_record() {
        return unique_function_record(new detail::function_record());
    }

    /// Special internal constructor for functors, lambda functions, etc.
    template <typename Func, typename Return, typename... Args, typename... Extra>
    void initialize(Func &&f, Return (*)(Args...), const Extra&... extra) {
        using namespace detail;
        struct capture { remove_reference_t<Func> f; };

        /* Store the function including any extra state it might have (e.g. a lambda capture object) */
        // The unique_ptr makes sure nothing is leaked in case of an exception.
        auto unique_rec = make_function_record();
        auto rec = unique_rec.get();

        /* Store the capture object directly in the function record if there is enough space */
        if (sizeof(capture) <= sizeof(rec->data)) {
            /* Without these pragmas, GCC warns that there might not be
               enough space to use the placement new operator. However, the
               'if' statement above ensures that this is the case. */
#if defined(__GNUG__) && !defined(__clang__) && __GNUC__ >= 6
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wplacement-new"
#endif
            new ((capture *) &rec->data) capture { std::forward<Func>(f) };
#if defined(__GNUG__) && !defined(__clang__) && __GNUC__ >= 6
#  pragma GCC diagnostic pop
#endif
            if (!std::is_trivially_destructible<Func>::value)
                rec->free_data = [](function_record *r) { ((capture *) &r->data)->~capture(); };
        } else {
            rec->data[0] = new capture { std::forward<Func>(f) };
            rec->free_data = [](function_record *r) { delete ((capture *) r->data[0]); };
        }

        /* Type casters for the function arguments and return value */
        using cast_in = argument_loader<Args...>;
        using cast_out = make_caster<
            conditional_t<std::is_void<Return>::value, void_type, Return>
        >;

        static_assert(expected_num_args<Extra...>(sizeof...(Args), cast_in::has_args, cast_in::has_kwargs),
                      "The number of argument annotations does not match the number of function arguments");

        /* Dispatch code which converts function arguments and performs the actual function call */
        rec->impl = [](function_call &call) -> handle {
            cast_in args_converter;

            /* Try to cast the function arguments into the C++ domain */
            if (!args_converter.load_args(call))
                return PYBIND11_TRY_NEXT_OVERLOAD;

            /* Invoke call policy pre-call hook */
            process_attributes<Extra...>::precall(call);

            /* Get a pointer to the capture object */
            auto data = (sizeof(capture) <= sizeof(call.func.data)
                         ? &call.func.data : call.func.data[0]);
            auto *cap = const_cast<capture *>(reinterpret_cast<const capture *>(data));

            /* Override policy for rvalues -- usually to enforce rvp::move on an rvalue */
            return_value_policy policy = return_value_policy_override<Return>::policy(call.func.policy);

            /* Function scope guard -- defaults to the compile-to-nothing `void_type` */
            using Guard = extract_guard_t<Extra...>;

            /* Perform the function call */
            handle result = cast_out::cast(
                std::move(args_converter).template call<Return, Guard>(cap->f), policy, call.parent);

            /* Invoke call policy post-call hook */
            process_attributes<Extra...>::postcall(call, result);

            return result;
        };

        /* Process any user-provided function attributes */
        process_attributes<Extra...>::init(extra..., rec);

        {
            constexpr bool has_kw_only_args = any_of<std::is_same<kw_only, Extra>...>::value,
                           has_pos_only_args = any_of<std::is_same<pos_only, Extra>...>::value,
                           has_args = any_of<std::is_same<args, Args>...>::value,
                           has_arg_annotations = any_of<is_keyword<Extra>...>::value;
            static_assert(has_arg_annotations || !has_kw_only_args, "py::kw_only requires the use of argument annotations");
            static_assert(has_arg_annotations || !has_pos_only_args, "py::pos_only requires the use of argument annotations (for docstrings and aligning the annotations to the argument)");
            static_assert(!(has_args && has_kw_only_args), "py::kw_only cannot be combined with a py::args argument");
        }

        /* Generate a readable signature describing the function's arguments and return value types */
        static constexpr auto signature = _("(") + cast_in::arg_names + _(") -> ") + cast_out::name;
        PYBIND11_DESCR_CONSTEXPR auto types = decltype(signature)::types();

        /* Register the function with Python from generic (non-templated) code */
        // Pass on the ownership over the `unique_rec` to `initialize_generic`. `rec` stays valid.
        initialize_generic(std::move(unique_rec), signature.text, types.data(), sizeof...(Args));

        if (cast_in::has_args) rec->has_args = true;
        if (cast_in::has_kwargs) rec->has_kwargs = true;

        /* Stash some additional information used by an important optimization in 'functional.h' */
        using FunctionType = Return (*)(Args...);
        constexpr bool is_function_ptr =
            std::is_convertible<Func, FunctionType>::value &&
            sizeof(capture) == sizeof(void *);
        if (is_function_ptr) {
            rec->is_stateless = true;
            rec->data[1] = const_cast<void *>(reinterpret_cast<const void *>(&typeid(FunctionType)));
        }
    }

    // Utility class that keeps track of all duplicated strings, and cleans them up in its destructor,
    // unless they are released. Basically a RAII-solution to deal with exceptions along the way.
    class strdup_guard {
    public:
        ~strdup_guard() {
            for (auto s : strings)
                std::free(s);
        }
        char *operator()(const char *s) {
            auto t = strdup(s);
            strings.push_back(t);
            return t;
        }
        void release() {
            strings.clear();
        }
    private:
        std::vector<char *> strings;
    };

    /// Register a function call with Python (generic non-templated code goes here)
    void initialize_generic(unique_function_record &&unique_rec, const char *text,
                            const std::type_info *const *types, size_t args) {
        // Do NOT receive `unique_rec` by value. If this function fails to move out the unique_ptr,
        // we do not want this to destuct the pointer. `initialize` (the caller) still relies on the
        // pointee being alive after this call. Only move out if a `capsule` is going to keep it alive.
        auto rec = unique_rec.get();

        // Keep track of strdup'ed strings, and clean them up as long as the function's capsule
        // has not taken ownership yet (when `unique_rec.release()` is called).
        // Note: This cannot easily be fixed by a `unique_ptr` with custom deleter, because the strings
        // are only referenced before strdup'ing. So only *after* the following block could `destruct`
        // safely be called, but even then, `repr` could still throw in the middle of copying all strings.
        strdup_guard guarded_strdup;

        /* Create copies of all referenced C-style strings */
        rec->name = guarded_strdup(rec->name ? rec->name : "");
        if (rec->doc) rec->doc = guarded_strdup(rec->doc);
        for (auto &a: rec->args) {
            if (a.name)
                a.name = guarded_strdup(a.name);
            if (a.descr)
                a.descr = guarded_strdup(a.descr);
            else if (a.value)
                a.descr = guarded_strdup(repr(a.value).cast<std::string>().c_str());
        }

        rec->is_constructor = !strcmp(rec->name, "__init__") || !strcmp(rec->name, "__setstate__");

#if !defined(NDEBUG) && !defined(PYBIND11_DISABLE_NEW_STYLE_INIT_WARNING)
        if (rec->is_constructor && !rec->is_new_style_constructor) {
            const auto class_name = detail::get_fully_qualified_tp_name((PyTypeObject *) rec->scope.ptr());
            const auto func_name = std::string(rec->name);
            PyErr_WarnEx(
                PyExc_FutureWarning,
                ("pybind11-bound class '" + class_name + "' is using an old-style "
                 "placement-new '" + func_name + "' which has been deprecated. See "
                 "the upgrade guide in pybind11's docs. This message is only visible "
                 "when compiled in debug mode.").c_str(), 0
            );
        }
#endif

#if PY_MAJOR_VERSION < 3
        if (strcmp(rec->name, "__next__") == 0) {
            std::free(rec->name);
            rec->name = guarded_strdup("next");
        } else if (strcmp(rec->name, "__bool__") == 0) {
            std::free(rec->name);
            rec->name = guarded_strdup("__nonzero__");
        }
#endif
        auto signature = detail::make_func_signature(rec, text, types, args);
        rec->signature = guarded_strdup(signature.c_str());
        rec->args.shrink_to_fit();
        rec->nargs = (std::uint16_t) args;

        if (rec->sibling && PYBIND11_INSTANCE_METHOD_CHECK(rec->sibling.ptr()))
            rec->sibling = PYBIND11_INSTANCE_METHOD_GET_FUNCTION(rec->sibling.ptr());

        detail::function_record *chain = nullptr, *chain_start = rec;
        if (rec->sibling) {
            if (PyCFunction_Check(rec->sibling.ptr())) {
                auto rec_capsule = reinterpret_borrow<capsule>(PyCFunction_GET_SELF(rec->sibling.ptr()));
                chain = (detail::function_record *) rec_capsule;
                /* Never append a method to an overload chain of a parent class;
                   instead, hide the parent's overloads in this case */
                if (!chain->scope.is(rec->scope))
                    chain = nullptr;
            }
            // Don't trigger for things like the default __init__, which are wrapper_descriptors that we are intentionally replacing
            else if (!rec->sibling.is_none() && rec->name[0] != '_')
                pybind11_fail("Cannot overload existing non-function object \"" + std::string(rec->name) +
                        "\" with a function of the same name");
        }

        if (!chain) {
            /* No existing overload was found, create a new function object */
            rec->def = new PyMethodDef();
            std::memset(rec->def, 0, sizeof(PyMethodDef));
            rec->def->ml_name = rec->name;
            rec->def->ml_meth = reinterpret_cast<PyCFunction>(reinterpret_cast<void (*) (void)>(*dispatcher));
            rec->def->ml_flags = METH_VARARGS | METH_KEYWORDS;

            capsule rec_capsule(unique_rec.release(), [](void *ptr) {
                destruct((detail::function_record *) ptr);
            });
            guarded_strdup.release();

            object scope_module;
            if (rec->scope) {
                if (hasattr(rec->scope, "__module__")) {
                    scope_module = rec->scope.attr("__module__");
                } else if (hasattr(rec->scope, "__name__")) {
                    scope_module = rec->scope.attr("__name__");
                }
            }

            m_ptr = PyCFunction_NewEx(rec->def, rec_capsule.ptr(), scope_module.ptr());
            if (!m_ptr)
                pybind11_fail("cpp_function::cpp_function(): Could not allocate function object");
        } else {
            /* Append at the beginning or end of the overload chain */
            m_ptr = rec->sibling.ptr();
            inc_ref();
            if (chain->is_method != rec->is_method)
                pybind11_fail("overloading a method with both static and instance methods is not supported; "
                    #if defined(NDEBUG)
                        "compile in debug mode for more details"
                    #else
                        "error while attempting to bind " + std::string(rec->is_method ? "instance" : "static") + " method " +
                        std::string(pybind11::str(rec->scope.attr("__name__"))) + "." + std::string(rec->name) + signature
                    #endif
                );

            if (rec->prepend) {
                // Beginning of chain; we need to replace the capsule's current head-of-the-chain
                // pointer with this one, then make this one point to the previous head of the
                // chain.
                chain_start = rec;
                rec->next = chain;
                auto rec_capsule = reinterpret_borrow<capsule>(((PyCFunctionObject *) m_ptr)->m_self);
                rec_capsule.set_pointer(unique_rec.release());
                guarded_strdup.release();
            } else {
                // Or end of chain (normal behavior)
                chain_start = chain;
                while (chain->next)
                    chain = chain->next;
                chain->next = unique_rec.release();
                guarded_strdup.release();
            }
        }

        auto doc = detail::make_func_doc(rec, chain != nullptr, chain_start);
        /* Install docstring */
        auto *func = (PyCFunctionObject *) m_ptr;
        std::free(const_cast<char *>(func->m_ml->ml_doc));
        // Install docstring if it's non-empty (when at least one option is enabled)
        func->m_ml->ml_doc = doc.empty() ? nullptr : strdup(doc.c_str());

        if (rec->is_method) {
            m_ptr = PYBIND11_INSTANCE_METHOD_NEW(m_ptr, rec->scope.ptr());
            if (!m_ptr)
                pybind11_fail("cpp_function::cpp_function(): Could not allocate instance method object");
            Py_DECREF(func);
        }
    }

    /// When a cpp_function is GCed, release any memory allocated by pybind11
    static void destruct(detail::function_record *rec, bool free_strings = true) {
        // If on Python 3.9, check the interpreter "MICRO" (patch) version.
        // If this is running on 3.9.0, we have to work around a bug.
        #if !defined(PYPY_VERSION) && PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION == 9
            static bool is_zero = Py_GetVersion()[4] == '0';
        #endif

        while (rec) {
            detail::function_record *next = rec->next;
            if (rec->free_data)
                rec->free_data(rec);
            // During initialization, these strings might not have been copied yet,
            // so they cannot be freed. Once the function has been created, they can.
            // Check `make_function_record` for more details.
            if (free_strings) {
                std::free((char *) rec->name);
                std::free((char *) rec->doc);
                std::free((char *) rec->signature);
                for (auto &arg: rec->args) {
                    std::free(const_cast<char *>(arg.name));
                    std::free(const_cast<char *>(arg.descr));
                }
            }
            for (auto &arg: rec->args)
                arg.value.dec_ref();
            if (rec->def) {
                std::free(const_cast<char *>(rec->def->ml_doc));
                // Python 3.9.0 decref's these in the wrong order; rec->def
                // If loaded on 3.9.0, let these leak (use Python 3.9.1 at runtime to fix)
                // See https://github.com/python/cpython/pull/22670
                #if !defined(PYPY_VERSION) && PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION == 9
                    if (!is_zero)
                        delete rec->def;
                #else
                    delete rec->def;
                #endif
            }
            delete rec;
            rec = next;
        }
    }

    /// Main dispatch logic for calls to functions bound using pybind11
    static PyObject *dispatcher(PyObject *self, PyObject *args_in, PyObject *kwargs_in) {
        using namespace detail;

        /* Iterator over the list of potentially admissible overloads */
        const function_record *overloads = (function_record *) PyCapsule_GetPointer(self, nullptr),
                              *it = overloads;

        /* Need to know how many arguments + keyword arguments there are to pick the right overload */
        const auto n_args_in = (size_t) PyTuple_GET_SIZE(args_in);

        handle parent = n_args_in > 0 ? PyTuple_GET_ITEM(args_in, 0) : nullptr,
               result = PYBIND11_TRY_NEXT_OVERLOAD;

        auto self_value_and_holder = value_and_holder();
        if (overloads->is_constructor) {
            if (!PyObject_TypeCheck(parent.ptr(), (PyTypeObject *) overloads->scope.ptr())) {
                PyErr_SetString(PyExc_TypeError, "__init__(self, ...) called with invalid `self` argument");
                return nullptr;
            }

            const auto tinfo = get_type_info((PyTypeObject *) overloads->scope.ptr());
            const auto pi = reinterpret_cast<instance *>(parent.ptr());
            self_value_and_holder = pi->get_value_and_holder(tinfo, true);

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
            std::vector<function_call> second_pass;

            // However, if there are no overloads, we can just skip the no-convert pass entirely
            const bool overloaded = it != nullptr && it->next != nullptr;

            for (; it != nullptr; it = it->next) {

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

                const function_record &func = *it;
                size_t num_args = func.nargs;    // Number of positional arguments that we need
                if (func.has_args) --num_args;   // (but don't count py::args
                if (func.has_kwargs) --num_args; //  or py::kwargs)
                size_t pos_args = num_args - func.nargs_kw_only;

                if (!func.has_args && n_args_in > pos_args)
                    continue; // Too many positional arguments for this overload

                if (n_args_in < pos_args && func.args.size() < pos_args)
                    continue; // Not enough positional arguments given, and not enough defaults to fill in the blanks

                function_call call(func, parent);

                size_t args_to_copy = (std::min)(pos_args, n_args_in); // Protect std::min with parentheses
                size_t args_copied = 0;

                // 0. Inject new-style `self` argument
                if (func.is_new_style_constructor) {
                    // The `value` may have been preallocated by an old-style `__init__`
                    // if it was a preceding candidate for overload resolution.
                    if (self_value_and_holder)
                        self_value_and_holder.type->dealloc(self_value_and_holder);

                    call.init_self = PyTuple_GET_ITEM(args_in, 0);
                    call.args.emplace_back(reinterpret_cast<PyObject *>(&self_value_and_holder));
                    call.args_convert.push_back(false);
                    ++args_copied;
                }

                // 1. Copy any position arguments given.
                bool bad_arg = false;
                for (; args_copied < args_to_copy; ++args_copied) {
                    const argument_record *arg_rec = args_copied < func.args.size() ? &func.args[args_copied] : nullptr;
                    if (kwargs_in && arg_rec && arg_rec->name && PyDict_GetItemString(kwargs_in, arg_rec->name)) {
                        bad_arg = true;
                        break;
                    }

                    handle arg(PyTuple_GET_ITEM(args_in, args_copied));
                    if (arg_rec && !arg_rec->none && arg.is_none()) {
                        bad_arg = true;
                        break;
                    }
                    call.args.push_back(arg);
                    call.args_convert.push_back(arg_rec ? arg_rec->convert : true);
                }
                if (bad_arg)
                    continue; // Maybe it was meant for another overload (issue #688)

                // We'll need to copy this if we steal some kwargs for defaults
                dict kwargs = reinterpret_borrow<dict>(kwargs_in);

                // 1.5. Fill in any missing pos_only args from defaults if they exist
                if (args_copied < func.nargs_pos_only) {
                    for (; args_copied < func.nargs_pos_only; ++args_copied) {
                        const auto &arg_rec = func.args[args_copied];
                        handle value;

                        if (arg_rec.value) {
                            value = arg_rec.value;
                        }
                        if (value) {
                            call.args.push_back(value);
                            call.args_convert.push_back(arg_rec.convert);
                        } else
                            break;
                    }

                    if (args_copied < func.nargs_pos_only)
                        continue; // Not enough defaults to fill the positional arguments
                }

                // 2. Check kwargs and, failing that, defaults that may help complete the list
                if (args_copied < num_args) {
                    bool copied_kwargs = false;

                    for (; args_copied < num_args; ++args_copied) {
                        const auto &arg_rec = func.args[args_copied];

                        handle value;
                        if (kwargs_in && arg_rec.name)
                            value = PyDict_GetItemString(kwargs.ptr(), arg_rec.name);

                        if (value) {
                            // Consume a kwargs value
                            if (!copied_kwargs) {
                                kwargs = reinterpret_steal<dict>(PyDict_Copy(kwargs.ptr()));
                                copied_kwargs = true;
                            }
                            PyDict_DelItemString(kwargs.ptr(), arg_rec.name);
                        } else if (arg_rec.value) {
                            value = arg_rec.value;
                        }

                        if (!arg_rec.none && value.is_none()) {
                            break;
                        }

                        if (value) {
                            call.args.push_back(value);
                            call.args_convert.push_back(arg_rec.convert);
                        }
                        else
                            break;
                    }

                    if (args_copied < num_args)
                        continue; // Not enough arguments, defaults, or kwargs to fill the positional arguments
                }

                // 3. Check everything was consumed (unless we have a kwargs arg)
                if (kwargs && !kwargs.empty() && !func.has_kwargs)
                    continue; // Unconsumed kwargs, but no py::kwargs argument to accept them

                // 4a. If we have a py::args argument, create a new tuple with leftovers
                if (func.has_args) {
                    tuple extra_args;
                    if (args_to_copy == 0) {
                        // We didn't copy out any position arguments from the args_in tuple, so we
                        // can reuse it directly without copying:
                        extra_args = reinterpret_borrow<tuple>(args_in);
                    } else if (args_copied >= n_args_in) {
                        extra_args = tuple(0);
                    } else {
                        size_t args_size = n_args_in - args_copied;
                        extra_args = tuple(args_size);
                        for (size_t i = 0; i < args_size; ++i) {
                            extra_args[i] = PyTuple_GET_ITEM(args_in, args_copied + i);
                        }
                    }
                    call.args.push_back(extra_args);
                    call.args_convert.push_back(false);
                    call.args_ref = std::move(extra_args);
                }

                // 4b. If we have a py::kwargs, pass on any remaining kwargs
                if (func.has_kwargs) {
                    if (!kwargs.ptr())
                        kwargs = dict(); // If we didn't get one, send an empty one
                    call.args.push_back(kwargs);
                    call.args_convert.push_back(false);
                    call.kwargs_ref = std::move(kwargs);
                }

                // 5. Put everything in a vector.  Not technically step 5, we've been building it
                // in `call.args` all along.
                #if !defined(NDEBUG)
                if (call.args.size() != func.nargs || call.args_convert.size() != func.nargs)
                    pybind11_fail("Internal error: function call dispatcher inserted wrong number of arguments!");
                #endif

                std::vector<bool> second_pass_convert;
                if (overloaded) {
                    // We're in the first no-convert pass, so swap out the conversion flags for a
                    // set of all-false flags.  If the call fails, we'll swap the flags back in for
                    // the conversion-allowed call below.
                    second_pass_convert.resize(func.nargs, false);
                    call.args_convert.swap(second_pass_convert);
                }

                // 6. Call the function.
                try {
                    loader_life_support guard{};
                    result = func.impl(call);
                } catch (reference_cast_error &) {
                    result = PYBIND11_TRY_NEXT_OVERLOAD;
                }

                if (result.ptr() != PYBIND11_TRY_NEXT_OVERLOAD)
                    break;

                if (overloaded) {
                    // The (overloaded) call failed; if the call has at least one argument that
                    // permits conversion (i.e. it hasn't been explicitly specified `.noconvert()`)
                    // then add this call to the list of second pass overloads to try.
                    for (size_t i = func.is_method ? 1 : 0; i < pos_args; i++) {
                        if (second_pass_convert[i]) {
                            // Found one: swap the converting flags back in and store the call for
                            // the second pass.
                            call.args_convert.swap(second_pass_convert);
                            second_pass.push_back(std::move(call));
                            break;
                        }
                    }
                }
            }

            if (overloaded && !second_pass.empty() && result.ptr() == PYBIND11_TRY_NEXT_OVERLOAD) {
                // The no-conversion pass finished without success, try again with conversion allowed
                for (auto &call : second_pass) {
                    try {
                        loader_life_support guard{};
                        result = call.func.impl(call);
                    } catch (reference_cast_error &) {
                        result = PYBIND11_TRY_NEXT_OVERLOAD;
                    }

                    if (result.ptr() != PYBIND11_TRY_NEXT_OVERLOAD) {
                        // The error reporting logic below expects 'it' to be valid, as it would be
                        // if we'd encountered this failure in the first-pass loop.
                        if (!result)
                            it = &call.func;
                        break;
                    }
                }
            }
        } catch (error_already_set &e) {
            e.restore();
            return nullptr;
#ifdef __GLIBCXX__
        } catch ( abi::__forced_unwind& ) {
            throw;
#endif
        } catch (...) {
            /* When an exception is caught, give each registered exception
               translator a chance to translate it to a Python exception
               in reverse order of registration.

               A translator may choose to do one of the following:

                - catch the exception and call PyErr_SetString or PyErr_SetObject
                  to set a standard (or custom) Python exception, or
                - do nothing and let the exception fall through to the next translator, or
                - delegate translation to the next translator by throwing a new type of exception. */

            auto last_exception = std::current_exception();
            auto &registered_exception_translators = get_internals().registered_exception_translators;
            for (auto& translator : registered_exception_translators) {
                try {
                    translator(last_exception);
                } catch (...) {
                    last_exception = std::current_exception();
                    continue;
                }
                return nullptr;
            }
            PyErr_SetString(PyExc_SystemError, "Exception escaped from default exception translator!");
            return nullptr;
        }

        auto append_note_if_missing_header_is_suspected = [](std::string &msg) {
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
            for (const function_record *it2 = overloads; it2 != nullptr; it2 = it2->next) {
                msg += "    "+ std::to_string(++ctr) + ". ";

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
                try {
                    msg += repr(args_[ti]);
                } catch (const error_already_set&) {
                    msg += "<repr raised Error>";
                }
            }
            if (kwargs_in) {
                auto kwargs = reinterpret_borrow<dict>(kwargs_in);
                if (!kwargs.empty()) {
                    if (some_args) msg += "; ";
                    msg += "kwargs: ";
                    bool first = true;
                    for (auto kwarg : kwargs) {
                        if (first) first = false;
                        else msg += ", ";
                        msg += pybind11::str("{}=").format(kwarg.first);
                        try {
                            msg += repr(kwarg.second);
                        } catch (const error_already_set&) {
                            msg += "<repr raised Error>";
                        }
                    }
                }
            }

            append_note_if_missing_header_is_suspected(msg);
            PyErr_SetString(PyExc_TypeError, msg.c_str());
            return nullptr;
        } else if (!result) {
            std::string msg = "Unable to convert function return value to a "
                              "Python type! The signature was\n\t";
            msg += it->signature;
            append_note_if_missing_header_is_suspected(msg);
            PyErr_SetString(PyExc_TypeError, msg.c_str());
            return nullptr;
        } else {
            if (overloads->is_constructor && !self_value_and_holder.holder_constructed()) {
                auto *pi = reinterpret_cast<instance *>(parent.ptr());
                self_value_and_holder.type->init_instance(pi, nullptr);
            }
            return result.ptr();
        }
    }
};
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
