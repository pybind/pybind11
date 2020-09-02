#include "pybind11.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

PYBIND11_INLINE cpp_function::cpp_function() {}

PYBIND11_INLINE cpp_function::cpp_function(std::nullptr_t) { }

PYBIND11_INLINE object cpp_function::name() const { return attr("__name__"); }

PYBIND11_INLINE detail::function_record *cpp_function::make_function_record() {
    return new detail::function_record();
}

PYBIND11_INLINE void cpp_function::initialize_generic(detail::function_record *rec, const char *text,
                        const std::type_info *const *types, size_t args) {

    /* Create copies of all referenced C-style strings */
    rec->name = strdup(rec->name ? rec->name : "");
    if (rec->doc) rec->doc = strdup(rec->doc);
    for (auto &a: rec->args) {
        if (a.name)
            a.name = strdup(a.name);
        if (a.descr)
            a.descr = strdup(a.descr);
        else if (a.value)
            a.descr = strdup(repr(a.value).cast<std::string>().c_str());
    }

    rec->is_constructor = !strcmp(rec->name, "__init__") || !strcmp(rec->name, "__setstate__");

#if !defined(NDEBUG) && !defined(PYBIND11_DISABLE_NEW_STYLE_INIT_WARNING)
    if (rec->is_constructor && !rec->is_new_style_constructor) {
        const auto class_name = std::string(((PyTypeObject *) rec->scope.ptr())->tp_name);
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

    /* Generate a proper function signature */
    std::string signature;
    size_t type_index = 0, arg_index = 0;
    for (auto *pc = text; *pc != '\0'; ++pc) {
        const auto c = *pc;

        if (c == '{') {
            // Write arg name for everything except *args and **kwargs.
            if (*(pc + 1) == '*')
                continue;

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

#if PY_MAJOR_VERSION < 3
    if (strcmp(rec->name, "__next__") == 0) {
        std::free(rec->name);
        rec->name = strdup("next");
    } else if (strcmp(rec->name, "__bool__") == 0) {
        std::free(rec->name);
        rec->name = strdup("__nonzero__");
    }
#endif
    rec->signature = strdup(signature.c_str());
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

        capsule rec_capsule(rec, [](void *ptr) {
            destruct((detail::function_record *) ptr);
        });

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
        /* Append at the end of the overload chain */
        m_ptr = rec->sibling.ptr();
        inc_ref();
        chain_start = chain;
        if (chain->is_method != rec->is_method)
            pybind11_fail("overloading a method with both static and instance methods is not supported; "
                #if defined(NDEBUG)
                    "compile in debug mode for more details"
                #else
                    "error while attempting to bind " + std::string(rec->is_method ? "instance" : "static") + " method " +
                    std::string(pybind11::str(rec->scope.attr("__name__"))) + "." + std::string(rec->name) + signature
                #endif
            );
        while (chain->next)
            chain = chain->next;
        chain->next = rec;
    }

    std::string signatures;
    int index = 0;
    /* Create a nice pydoc rec including all signatures and
        docstrings of the functions in the overload chain */
    if (chain && options::show_function_signatures()) {
        // First a generic signature
        signatures += rec->name;
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
            signatures += rec->name;
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
    PyCFunctionObject *func = (PyCFunctionObject *) m_ptr;
    if (func->m_ml->ml_doc)
        std::free(const_cast<char *>(func->m_ml->ml_doc));
    func->m_ml->ml_doc = strdup(signatures.c_str());

    if (rec->is_method) {
        m_ptr = PYBIND11_INSTANCE_METHOD_NEW(m_ptr, rec->scope.ptr());
        if (!m_ptr)
            pybind11_fail("cpp_function::cpp_function(): Could not allocate instance method object");
        Py_DECREF(func);
    }
}

PYBIND11_INLINE void cpp_function::destruct(detail::function_record *rec) {
    while (rec) {
        detail::function_record *next = rec->next;
        if (rec->free_data)
            rec->free_data(rec);
        std::free((char *) rec->name);
        std::free((char *) rec->doc);
        std::free((char *) rec->signature);
        for (auto &arg: rec->args) {
            std::free(const_cast<char *>(arg.name));
            std::free(const_cast<char *>(arg.descr));
            arg.value.dec_ref();
        }
        if (rec->def) {
            std::free(const_cast<char *>(rec->def->ml_doc));
            delete rec->def;
        }
        delete rec;
        rec = next;
    }
}

PYBIND11_INLINE PyObject *cpp_function::dispatcher(PyObject *self, PyObject *args_in, PyObject *kwargs_in) {
    using namespace detail;

    /* Iterator over the list of potentially admissible overloads */
    const function_record *overloads = (function_record *) PyCapsule_GetPointer(self, nullptr),
                            *it = overloads;

    /* Need to know how many arguments + keyword arguments there are to pick the right overload */
    const size_t n_args_in = (size_t) PyTuple_GET_SIZE(args_in);

    handle parent = n_args_in > 0 ? PyTuple_GET_ITEM(args_in, 0) : nullptr,
            result = PYBIND11_TRY_NEXT_OVERLOAD;

    auto self_value_and_holder = value_and_holder();
    if (overloads->is_constructor) {
        const auto tinfo = get_type_info((PyTypeObject *) overloads->scope.ptr());
        const auto pi = reinterpret_cast<instance *>(parent.ptr());
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
            size_t pos_args = num_args - func.nargs_kwonly;

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
                call.args.push_back(reinterpret_cast<PyObject *>(&self_value_and_holder));
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

            // 2. Check kwargs and, failing that, defaults that may help complete the list
            if (args_copied < num_args) {
                bool copied_kwargs = false;

                for (; args_copied < num_args; ++args_copied) {
                    const auto &arg = func.args[args_copied];

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
                    } else if (arg.value) {
                        value = arg.value;
                    }

                    if (value) {
                        call.args.push_back(value);
                        call.args_convert.push_back(arg.convert);
                    }
                    else
                        break;
                }

                if (args_copied < num_args)
                    continue; // Not enough arguments, defaults, or kwargs to fill the positional arguments
            }

            // 3. Check everything was consumed (unless we have a kwargs arg)
            if (kwargs && kwargs.size() > 0 && !func.has_kwargs)
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
#if defined(__GNUG__) && !defined(__clang__)
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
                msg += pybind11::repr(args_[ti]);
            } catch (const error_already_set&) {
                msg += "<repr raised Error>";
            }
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
                    msg += pybind11::str("{}=").format(kwarg.first);
                    try {
                        msg += pybind11::repr(kwarg.second);
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

PYBIND11_INLINE module::module(const char *name, const char *doc) {
    if (!options::show_user_defined_docstrings()) doc = nullptr;
#if PY_MAJOR_VERSION >= 3
    PyModuleDef *def = new PyModuleDef();
    std::memset(def, 0, sizeof(PyModuleDef));
    def->m_name = name;
    def->m_doc = doc;
    def->m_size = -1;
    Py_INCREF(def);
    m_ptr = PyModule_Create(def);
#else
    m_ptr = Py_InitModule3(name, nullptr, doc);
#endif
    if (m_ptr == nullptr)
        pybind11_fail("Internal error in module::module()");
    inc_ref();
}

PYBIND11_INLINE module module::def_submodule(const char *name, const char *doc) {
    std::string full_name = std::string(PyModule_GetName(m_ptr))
        + std::string(".") + std::string(name);
    auto result = reinterpret_borrow<module>(PyImport_AddModule(full_name.c_str()));
    if (doc && options::show_user_defined_docstrings())
        result.attr("__doc__") = pybind11::str(doc);
    attr(name) = result;
    return result;
}

PYBIND11_INLINE module module::import(const char *name) {
    PyObject *obj = PyImport_ImportModule(name);
    if (!obj)
        throw error_already_set();
    return reinterpret_steal<module>(obj);
}

PYBIND11_INLINE void module::reload() {
    PyObject *obj = PyImport_ReloadModule(ptr());
    if (!obj)
        throw error_already_set();
    *this = reinterpret_steal<module>(obj);
}

PYBIND11_INLINE void module::add_object(const char *name, handle obj, bool overwrite) {
    if (!overwrite && hasattr(*this, name))
        pybind11_fail("Error during initialization: multiple incompatible definitions with name \"" +
                std::string(name) + "\"");

    PyModule_AddObject(ptr(), name, obj.inc_ref().ptr() /* steals a reference */);
}

PYBIND11_INLINE dict globals() {
    PyObject *p = PyEval_GetGlobals();
    return reinterpret_borrow<dict>(p ? p : module::import("__main__").attr("__dict__").ptr());
}

PYBIND11_NAMESPACE_BEGIN(detail)

PYBIND11_INLINE void generic_type::initialize(const type_record &rec) {
    if (rec.scope && hasattr(rec.scope, rec.name))
        pybind11_fail("generic_type: cannot initialize type \"" + std::string(rec.name) +
                        "\": an object with that name is already defined");

    if (rec.module_local ? get_local_type_info(*rec.type) : get_global_type_info(*rec.type))
        pybind11_fail("generic_type: type \"" + std::string(rec.name) +
                        "\" is already registered!");

    m_ptr = make_new_python_type(rec);

    /* Register supplemental type information in C++ dict */
    auto *tinfo = new detail::type_info();
    tinfo->type = (PyTypeObject *) m_ptr;
    tinfo->cpptype = rec.type;
    tinfo->type_size = rec.type_size;
    tinfo->type_align = rec.type_align;
    tinfo->operator_new = rec.operator_new;
    tinfo->holder_size_in_ptrs = size_in_ptrs(rec.holder_size);
    tinfo->init_instance = rec.init_instance;
    tinfo->dealloc = rec.dealloc;
    tinfo->simple_type = true;
    tinfo->simple_ancestors = true;
    tinfo->default_holder = rec.default_holder;
    tinfo->module_local = rec.module_local;

    auto &internals = get_internals();
    auto tindex = std::type_index(*rec.type);
    tinfo->direct_conversions = &internals.direct_conversions[tindex];
    if (rec.module_local)
        registered_local_types_cpp()[tindex] = tinfo;
    else
        internals.registered_types_cpp[tindex] = tinfo;
    internals.registered_types_py[(PyTypeObject *) m_ptr] = { tinfo };

    if (rec.bases.size() > 1 || rec.multiple_inheritance) {
        mark_parents_nonsimple(tinfo->type);
        tinfo->simple_ancestors = false;
    }
    else if (rec.bases.size() == 1) {
        auto parent_tinfo = get_type_info((PyTypeObject *) rec.bases[0].ptr());
        tinfo->simple_ancestors = parent_tinfo->simple_ancestors;
    }

    if (rec.module_local) {
        // Stash the local typeinfo and loader so that external modules can access it.
        tinfo->module_local_load = &type_caster_generic::local_load;
        setattr(m_ptr, PYBIND11_MODULE_LOCAL_ID, capsule(tinfo));
    }
}

PYBIND11_INLINE void generic_type::mark_parents_nonsimple(PyTypeObject *value) {
    auto t = reinterpret_borrow<tuple>(value->tp_bases);
    for (handle h : t) {
        auto tinfo2 = get_type_info((PyTypeObject *) h.ptr());
        if (tinfo2)
            tinfo2->simple_type = false;
        mark_parents_nonsimple((PyTypeObject *) h.ptr());
    }
}

PYBIND11_INLINE void generic_type::install_buffer_funcs(
        buffer_info *(*get_buffer)(PyObject *, void *),
        void *get_buffer_data) {
    PyHeapTypeObject *type = (PyHeapTypeObject*) m_ptr;
    auto tinfo = detail::get_type_info(&type->ht_type);

    if (!type->ht_type.tp_as_buffer)
        pybind11_fail(
            "To be able to register buffer protocol support for the type '" +
            std::string(tinfo->type->tp_name) +
            "' the associated class<>(..) invocation must "
            "include the pybind11::buffer_protocol() annotation!");

    tinfo->get_buffer = get_buffer;
    tinfo->get_buffer_data = get_buffer_data;
}

PYBIND11_INLINE void generic_type::def_property_static_impl(const char *name,
                                handle fget, handle fset,
                                detail::function_record *rec_func) {
    const auto is_static = rec_func && !(rec_func->is_method && rec_func->scope);
    const auto has_doc = rec_func && rec_func->doc && pybind11::options::show_user_defined_docstrings();
    auto property = handle((PyObject *) (is_static ? get_internals().static_property_type
                                                    : &PyProperty_Type));
    attr(name) = property(fget.ptr() ? fget : none(),
                            fset.ptr() ? fset : none(),
                            /*deleter*/none(),
                            pybind11::str(has_doc ? rec_func->doc : ""));
}

PYBIND11_INLINE void call_operator_delete(void *p, size_t s, size_t a) {
    (void)s; (void)a;
    #if defined(__cpp_aligned_new) && (!defined(_MSC_VER) || _MSC_VER >= 1912)
        if (a > __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
            #ifdef __cpp_sized_deallocation
                ::operator delete(p, s, std::align_val_t(a));
            #else
                ::operator delete(p, std::align_val_t(a));
            #endif
            return;
        }
    #endif
    #ifdef __cpp_sized_deallocation
        ::operator delete(p, s);
    #else
        ::operator delete(p);
    #endif
}

PYBIND11_INLINE void add_class_method(object& cls, const char *name_, const cpp_function &cf) {
    cls.attr(cf.name()) = cf;
    if (strcmp(name_, "__eq__") == 0 && !cls.attr("__dict__").contains("__hash__")) {
      cls.attr("__hash__") = none();
    }
}

PYBIND11_INLINE enum_base::enum_base(handle base, handle parent) : m_base(base), m_parent(parent) { }

PYBIND11_INLINE void enum_base::init(bool is_arithmetic, bool is_convertible) {
    m_base.attr("__entries") = dict();
    auto property = handle((PyObject *) &PyProperty_Type);
    auto static_property = handle((PyObject *) get_internals().static_property_type);

    m_base.attr("__repr__") = cpp_function(
        [](handle arg) -> str {
            handle type = arg.get_type();
            object type_name = type.attr("__name__");
            dict entries = type.attr("__entries");
            for (const auto &kv : entries) {
                object other = kv.second[int_(0)];
                if (other.equal(arg))
                    return pybind11::str("{}.{}").format(type_name, kv.first);
            }
            return pybind11::str("{}.???").format(type_name);
        }, name("__repr__"), is_method(m_base)
    );

    m_base.attr("name") = property(cpp_function(
        [](handle arg) -> str {
            dict entries = arg.get_type().attr("__entries");
            for (const auto &kv : entries) {
                if (handle(kv.second[int_(0)]).equal(arg))
                    return pybind11::str(kv.first);
            }
            return "???";
        }, name("name"), is_method(m_base)
    ));

    m_base.attr("__doc__") = static_property(cpp_function(
        [](handle arg) -> std::string {
            std::string docstring;
            dict entries = arg.attr("__entries");
            if (((PyTypeObject *) arg.ptr())->tp_doc)
                docstring += std::string(((PyTypeObject *) arg.ptr())->tp_doc) + "\n\n";
            docstring += "Members:";
            for (const auto &kv : entries) {
                auto key = std::string(pybind11::str(kv.first));
                auto comment = kv.second[int_(1)];
                docstring += "\n\n  " + key;
                if (!comment.is_none())
                    docstring += " : " + (std::string) pybind11::str(comment);
            }
            return docstring;
        }, name("__doc__")
    ), none(), none(), "");

    m_base.attr("__members__") = static_property(cpp_function(
        [](handle arg) -> dict {
            dict entries = arg.attr("__entries"), m;
            for (const auto &kv : entries)
                m[kv.first] = kv.second[int_(0)];
            return m;
        }, name("__members__")), none(), none(), ""
    );

    #define PYBIND11_ENUM_OP_STRICT(op, expr, strict_behavior)                     \
        m_base.attr(op) = cpp_function(                                            \
            [](object a, object b) {                                               \
                if (!a.get_type().is(b.get_type()))                                \
                    strict_behavior;                                               \
                return expr;                                                       \
            },                                                                     \
            name(op), is_method(m_base))

    #define PYBIND11_ENUM_OP_CONV(op, expr)                                        \
        m_base.attr(op) = cpp_function(                                            \
            [](object a_, object b_) {                                             \
                int_ a(a_), b(b_);                                                 \
                return expr;                                                       \
            },                                                                     \
            name(op), is_method(m_base))

    #define PYBIND11_ENUM_OP_CONV_LHS(op, expr)                                    \
        m_base.attr(op) = cpp_function(                                            \
            [](object a_, object b) {                                              \
                int_ a(a_);                                                        \
                return expr;                                                       \
            },                                                                     \
            name(op), is_method(m_base))

    if (is_convertible) {
        PYBIND11_ENUM_OP_CONV_LHS("__eq__", !b.is_none() &&  a.equal(b));
        PYBIND11_ENUM_OP_CONV_LHS("__ne__",  b.is_none() || !a.equal(b));

        if (is_arithmetic) {
            PYBIND11_ENUM_OP_CONV("__lt__",   a <  b);
            PYBIND11_ENUM_OP_CONV("__gt__",   a >  b);
            PYBIND11_ENUM_OP_CONV("__le__",   a <= b);
            PYBIND11_ENUM_OP_CONV("__ge__",   a >= b);
            PYBIND11_ENUM_OP_CONV("__and__",  a &  b);
            PYBIND11_ENUM_OP_CONV("__rand__", a &  b);
            PYBIND11_ENUM_OP_CONV("__or__",   a |  b);
            PYBIND11_ENUM_OP_CONV("__ror__",  a |  b);
            PYBIND11_ENUM_OP_CONV("__xor__",  a ^  b);
            PYBIND11_ENUM_OP_CONV("__rxor__", a ^  b);
            m_base.attr("__invert__") = cpp_function(
                [](object arg) { return ~(int_(arg)); }, name("__invert__"), is_method(m_base));
        }
    } else {
        PYBIND11_ENUM_OP_STRICT("__eq__",  int_(a).equal(int_(b)), return false);
        PYBIND11_ENUM_OP_STRICT("__ne__", !int_(a).equal(int_(b)), return true);

        if (is_arithmetic) {
            #define PYBIND11_THROW throw type_error("Expected an enumeration of matching type!");
            PYBIND11_ENUM_OP_STRICT("__lt__", int_(a) <  int_(b), PYBIND11_THROW);
            PYBIND11_ENUM_OP_STRICT("__gt__", int_(a) >  int_(b), PYBIND11_THROW);
            PYBIND11_ENUM_OP_STRICT("__le__", int_(a) <= int_(b), PYBIND11_THROW);
            PYBIND11_ENUM_OP_STRICT("__ge__", int_(a) >= int_(b), PYBIND11_THROW);
            #undef PYBIND11_THROW
        }
    }

    #undef PYBIND11_ENUM_OP_CONV_LHS
    #undef PYBIND11_ENUM_OP_CONV
    #undef PYBIND11_ENUM_OP_STRICT

    m_base.attr("__getstate__") = cpp_function(
        [](object arg) { return int_(arg); }, name("__getstate__"), is_method(m_base));

    m_base.attr("__hash__") = cpp_function(
        [](object arg) { return int_(arg); }, name("__hash__"), is_method(m_base));
}

PYBIND11_INLINE void enum_base::value(char const* name_, object value, const char *doc) {
    dict entries = m_base.attr("__entries");
    str name(name_);
    if (entries.contains(name)) {
        std::string type_name = (std::string) str(m_base.attr("__name__"));
        throw value_error(type_name + ": element \"" + std::string(name_) + "\" already exists!");
    }

    entries[name] = std::make_pair(value, doc);
    m_base.attr(name) = value;
}

PYBIND11_INLINE void enum_base::export_values() {
    dict entries = m_base.attr("__entries");
    for (const auto &kv : entries)
        m_parent.attr(kv.first) = kv.second[int_(0)];
}

PYBIND11_INLINE void keep_alive_impl(handle nurse, handle patient) {
    if (!nurse || !patient)
        pybind11_fail("Could not activate keep_alive!");

    if (patient.is_none() || nurse.is_none())
        return; /* Nothing to keep alive or nothing to be kept alive by */

    auto tinfo = all_type_info(Py_TYPE(nurse.ptr()));
    if (!tinfo.empty()) {
        /* It's a pybind-registered type, so we can store the patient in the
         * internal list. */
        add_patient(nurse.ptr(), patient.ptr());
    }
    else {
        /* Fall back to clever approach based on weak references taken from
         * Boost.Python. This is not used for pybind-registered types because
         * the objects can be destroyed out-of-order in a GC pass. */
        cpp_function disable_lifesupport(
            [patient](handle weakref) { patient.dec_ref(); weakref.dec_ref(); });

        weakref wr(nurse, disable_lifesupport);

        patient.inc_ref(); /* reference patient and leak the weak reference */
        (void) wr.release();
    }
}

PYBIND11_INLINE void keep_alive_impl(size_t Nurse, size_t Patient, function_call &call, handle ret) {
    auto get_arg = [&](size_t n) {
        if (n == 0)
            return ret;
        else if (n == 1 && call.init_self)
            return call.init_self;
        else if (n <= call.args.size())
            return call.args[n - 1];
        return handle();
    };

    keep_alive_impl(get_arg(Nurse), get_arg(Patient));
}

PYBIND11_INLINE std::pair<decltype(internals::registered_types_py)::iterator, bool> all_type_info_get_cache(PyTypeObject *type) {
    auto res = get_internals().registered_types_py
#ifdef __cpp_lib_unordered_map_try_emplace
        .try_emplace(type);
#else
        .emplace(type, std::vector<detail::type_info *>());
#endif
    if (res.second) {
        // New cache entry created; set up a weak reference to automatically remove it if the type
        // gets destroyed:
        weakref((PyObject *) type, cpp_function([type](handle wr) {
            get_internals().registered_types_py.erase(type);
            wr.dec_ref();
        })).release();
    }

    return res;
}

PYBIND11_INLINE void print(tuple args, dict kwargs) {
    auto strings = tuple(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
        strings[i] = str(args[i]);
    }
    auto sep = kwargs.contains("sep") ? kwargs["sep"] : cast(" ");
    auto line = sep.attr("join")(strings);

    object file;
    if (kwargs.contains("file")) {
        file = kwargs["file"].cast<object>();
    } else {
        try {
            file = module::import("sys").attr("stdout");
        } catch (const error_already_set &) {
            /* If print() is called from code that is executed as
               part of garbage collection during interpreter shutdown,
               importing 'sys' can fail. Give up rather than crashing the
               interpreter in this case. */
            return;
        }
    }

    auto write = file.attr("write");
    write(line);
    write(kwargs.contains("end") ? kwargs["end"] : cast("\n"));

    if (kwargs.contains("flush") && kwargs["flush"].cast<bool>())
        file.attr("flush")();
}

PYBIND11_NAMESPACE_END(detail)

#if defined(WITH_THREAD) && !defined(PYPY_VERSION)

PYBIND11_INLINE gil_scoped_acquire::gil_scoped_acquire() {
    auto const &internals = detail::get_internals();
    tstate = (PyThreadState *) PYBIND11_TLS_GET_VALUE(internals.tstate);

    if (!tstate) {
        /* Check if the GIL was acquired using the PyGILState_* API instead (e.g. if
            calling from a Python thread). Since we use a different key, this ensures
            we don't create a new thread state and deadlock in PyEval_AcquireThread
            below. Note we don't save this state with internals.tstate, since we don't
            create it we would fail to clear it (its reference count should be > 0). */
        tstate = PyGILState_GetThisThreadState();
    }

    if (!tstate) {
        tstate = PyThreadState_New(internals.istate);
        #if !defined(NDEBUG)
            if (!tstate)
                pybind11_fail("scoped_acquire: could not create thread state!");
        #endif
        tstate->gilstate_counter = 0;
        PYBIND11_TLS_REPLACE_VALUE(internals.tstate, tstate);
    } else {
        release = detail::get_thread_state_unchecked() != tstate;
    }

    if (release) {
        /* Work around an annoying assertion in PyThreadState_Swap */
        #if defined(Py_DEBUG)
            PyInterpreterState *interp = tstate->interp;
            tstate->interp = nullptr;
        #endif
        PyEval_AcquireThread(tstate);
        #if defined(Py_DEBUG)
            tstate->interp = interp;
        #endif
    }

    inc_ref();
}

PYBIND11_INLINE void gil_scoped_acquire::inc_ref() {
    ++tstate->gilstate_counter;
}

PYBIND11_INLINE void gil_scoped_acquire::dec_ref() {
    --tstate->gilstate_counter;
    #if !defined(NDEBUG)
        if (detail::get_thread_state_unchecked() != tstate)
            pybind11_fail("scoped_acquire::dec_ref(): thread state must be current!");
        if (tstate->gilstate_counter < 0)
            pybind11_fail("scoped_acquire::dec_ref(): reference count underflow!");
    #endif
    if (tstate->gilstate_counter == 0) {
        #if !defined(NDEBUG)
            if (!release)
                pybind11_fail("scoped_acquire::dec_ref(): internal error!");
        #endif
        PyThreadState_Clear(tstate);
        PyThreadState_DeleteCurrent();
        PYBIND11_TLS_DELETE_VALUE(detail::get_internals().tstate);
        release = false;
    }
}

PYBIND11_INLINE gil_scoped_acquire::~gil_scoped_acquire() {
    dec_ref();
    if (release)
        PyEval_SaveThread();
}

PYBIND11_INLINE gil_scoped_release::gil_scoped_release(bool disassoc) : disassoc(disassoc) {
    // `get_internals()` must be called here unconditionally in order to initialize
    // `internals.tstate` for subsequent `gil_scoped_acquire` calls. Otherwise, an
    // initialization race could occur as multiple threads try `gil_scoped_acquire`.
    const auto &internals = detail::get_internals();
    tstate = PyEval_SaveThread();
    if (disassoc) {
        auto key = internals.tstate;
        PYBIND11_TLS_DELETE_VALUE(key);
    }
}

PYBIND11_INLINE gil_scoped_release::~gil_scoped_release() {
    if (!tstate)
        return;
    PyEval_RestoreThread(tstate);
    if (disassoc) {
        auto key = detail::get_internals().tstate;
        PYBIND11_TLS_REPLACE_VALUE(key, tstate);
    }
}

#elif defined(PYPY_VERSION)
PYBIND11_INLINE gil_scoped_acquire::gil_scoped_acquire() { state = PyGILState_Ensure(); }
PYBIND11_INLINE gil_scoped_acquire::~gil_scoped_acquire() { PyGILState_Release(state); }
PYBIND11_INLINE gil_scoped_release::gil_scoped_release() { state = PyEval_SaveThread(); }
PYBIND11_INLINE gil_scoped_release::~gil_scoped_release() { PyEval_RestoreThread(state); }
#else
#endif

PYBIND11_INLINE error_already_set::~error_already_set() {
    if (m_type) {
        gil_scoped_acquire gil;
        error_scope scope;
        m_type.release().dec_ref();
        m_value.release().dec_ref();
        m_trace.release().dec_ref();
    }
}

PYBIND11_INLINE function get_type_overload(const void *this_ptr, const detail::type_info *this_type, const char *name)  {
    handle self = detail::get_object_handle(this_ptr, this_type);
    if (!self)
        return function();
    handle type = self.get_type();
    auto key = std::make_pair(type.ptr(), name);

    /* Cache functions that aren't overloaded in Python to avoid
       many costly Python dictionary lookups below */
    auto &cache = detail::get_internals().inactive_overload_cache;
    if (cache.find(key) != cache.end())
        return function();

    function overload = getattr(self, name, function());
    if (overload.is_cpp_function()) {
        cache.insert(key);
        return function();
    }

    /* Don't call dispatch code if invoked from overridden function.
       Unfortunately this doesn't work on PyPy. */
#if !defined(PYPY_VERSION)
    PyFrameObject *frame = PyThreadState_Get()->frame;
    if (frame && (std::string) str(frame->f_code->co_name) == name &&
        frame->f_code->co_argcount > 0) {
        PyFrame_FastToLocals(frame);
        PyObject *self_caller = PyDict_GetItem(
            frame->f_locals, PyTuple_GET_ITEM(frame->f_code->co_varnames, 0));
        if (self_caller == self.ptr())
            return function();
    }
#else
    /* PyPy currently doesn't provide a detailed cpyext emulation of
       frame objects, so we have to emulate this using Python. This
       is going to be slow..*/
    dict d; d["self"] = self; d["name"] = pybind11::str(name);
    PyObject *result = PyRun_String(
        "import inspect\n"
        "frame = inspect.currentframe()\n"
        "if frame is not None:\n"
        "    frame = frame.f_back\n"
        "    if frame is not None and str(frame.f_code.co_name) == name and "
        "frame.f_code.co_argcount > 0:\n"
        "        self_caller = frame.f_locals[frame.f_code.co_varnames[0]]\n"
        "        if self_caller == self:\n"
        "            self = None\n",
        Py_file_input, d.ptr(), d.ptr());
    if (result == nullptr)
        throw error_already_set();
    if (d["self"].is_none())
        return function();
    Py_DECREF(result);
#endif

    return overload;
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
