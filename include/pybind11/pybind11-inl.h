/*
    pybind11/pybind11-inl.h: Out-of-line definitions for pybind11.h

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

// Every function defined here must start with PYBIND11_INLINE (or
// PYBIND11_NOINLINE_ATTR PYBIND11_INLINE). In the default header-only mode this file is
// included at the bottom of pybind11.h; when PYBIND11_PRECOMPILED is defined it is only
// compiled into the pybind11 static library (see src/).

#pragma once

#include "pybind11.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)
PYBIND11_INLINE std::string replace_newlines_and_squash(const char *text) {
    const char *whitespaces = " \t\n\r\f\v";
    std::string result(text);
    bool previous_is_whitespace = false;

    if (result.size() >= 2) {
        // Do not modify string representations
        char first_char = result[0];
        char last_char = result[result.size() - 1];
        if (first_char == last_char && first_char == '\'') {
            return result;
        }
    }
    result.clear();

    // Replace characters in whitespaces array with spaces and squash consecutive spaces
    while (*text != '\0') {
        if (std::strchr(whitespaces, *text)) {
            if (!previous_is_whitespace) {
                result += ' ';
                previous_is_whitespace = true;
            }
        } else {
            result += *text;
            previous_is_whitespace = false;
        }
        ++text;
    }

    // Strip leading and trailing whitespaces
    const size_t str_begin = result.find_first_not_of(whitespaces);
    if (str_begin == std::string::npos) {
        return "";
    }

    const size_t str_end = result.find_last_not_of(whitespaces);
    const size_t str_range = str_end - str_begin + 1;

    return result.substr(str_begin, str_range);
}

PYBIND11_INLINE std::string generate_function_signature(const char *type_caster_name_field,
                                               detail::function_record *func_rec,
                                               const std::type_info *const *types,
                                               size_t &type_index,
                                               size_t &arg_index) {
    std::string signature;
    bool is_starred = false;
    // `is_return_value.top()` is true if we are currently inside the return type of the
    // signature. Using `@^`/`@$` we can force types to be arg/return types while `@!` pops
    // back to the previous state.
    std::stack<bool> is_return_value({false});
    // The following characters have special meaning in the signature parsing. Literals
    // containing these are escaped with `!`.
    std::string special_chars("!@%{}-");
    for (const auto *pc = type_caster_name_field; *pc != '\0'; ++pc) {
        const auto c = *pc;
        if (c == '{') {
            // Write arg name for everything except *args and **kwargs.
            // Detect {@*args...} or {@**kwargs...}
            is_starred = *(pc + 1) == '@' && *(pc + 2) == '*';
            if (is_starred) {
                continue;
            }
            // Separator for keyword-only arguments, placed before the kw
            // arguments start (unless we are already putting an *args)
            if (!func_rec->has_args && arg_index == func_rec->nargs_pos) {
                signature += "*, ";
            }
            if (arg_index < func_rec->args.size() && func_rec->args[arg_index].name) {
                signature += func_rec->args[arg_index].name;
            } else if (arg_index == 0 && func_rec->is_method) {
                signature += "self";
            } else {
                signature += "arg" + std::to_string(arg_index - (func_rec->is_method ? 1 : 0));
            }
            signature += ": ";
        } else if (c == '}') {
            // Write default value if available.
            if (!is_starred && arg_index < func_rec->args.size()
                && func_rec->args[arg_index].descr) {
                signature += " = ";
                signature += detail::replace_newlines_and_squash(func_rec->args[arg_index].descr);
            }
            // Separator for positional-only arguments (placed after the
            // argument, rather than before like *
            if (func_rec->nargs_pos_only > 0 && (arg_index + 1) == func_rec->nargs_pos_only) {
                signature += ", /";
            }
            if (!is_starred) {
                arg_index++;
            }
        } else if (c == '%') {
            const std::type_info *t = types[type_index++];
            if (!t) {
                pybind11_fail("Internal error while parsing type signature (1)");
            }
            if (auto *tinfo = detail::get_type_info(*t)) {
                handle th(reinterpret_cast<PyObject *>(tinfo->type));
                signature += th.attr("__module__").cast<std::string>() + "."
                             + th.attr("__qualname__").cast<std::string>();
            } else if (auto th = detail::global_internals_native_enum_type_map_get_item(*t)) {
                signature += th.attr("__module__").cast<std::string>() + "."
                             + th.attr("__qualname__").cast<std::string>();
            } else if (func_rec->is_new_style_constructor && arg_index == 0) {
                // A new-style `__init__` takes `self` as `value_and_holder`.
                // Rewrite it to the proper class type.
                signature += func_rec->scope.attr("__module__").cast<std::string>() + "."
                             + func_rec->scope.attr("__qualname__").cast<std::string>();
            } else {
                signature += detail::quote_cpp_type_name(detail::clean_type_id(t->name()));
            }
        } else if (c == '!' && special_chars.find(*(pc + 1)) != std::string::npos) {
            // typing::Literal escapes special characters with !
            signature += *++pc;
        } else if (c == '@') {
            // `@^ ... @!` and `@$ ... @!` are used to force arg/return value type (see
            // typing::Callable/detail::arg_descr/detail::return_descr).
            // `@~ ... @!` inverts the current context (see detail::inv_descr).
            if (*(pc + 1) == '^') {
                is_return_value.emplace(false);
                ++pc;
                continue;
            }
            if (*(pc + 1) == '$') {
                is_return_value.emplace(true);
                ++pc;
                continue;
            }
            if (*(pc + 1) == '~') {
                is_return_value.emplace(!is_return_value.top());
                ++pc;
                continue;
            }
            if (*(pc + 1) == '!') {
                is_return_value.pop();
                ++pc;
                continue;
            }
            // Handle types that differ depending on whether they appear
            // in an argument or a return value position (see io_name<text1, text2>).
            // For named arguments (py::arg()) with noconvert set, return value type is used.
            ++pc;
            if (!is_return_value.top()
                && (!(arg_index < func_rec->args.size() && !func_rec->args[arg_index].convert))) {
                while (*pc != '\0' && *pc != '@') {
                    signature += *pc++;
                }
                if (*pc == '@') {
                    ++pc;
                }
                while (*pc != '\0' && *pc != '@') {
                    ++pc;
                }
            } else {
                while (*pc != '\0' && *pc != '@') {
                    ++pc;
                }
                if (*pc == '@') {
                    ++pc;
                }
                while (*pc != '\0' && *pc != '@') {
                    signature += *pc++;
                }
            }
        } else {
            if (c == '-' && *(pc + 1) == '>') {
                is_return_value.emplace(true);
            }
            signature += c;
        }
    }
    return signature;
}

PYBIND11_NAMESPACE_BEGIN(function_record_PyTypeObject_methods)
PYBIND11_INLINE void tp_dealloc_impl(PyObject *self) {
    // Save type before PyObject_Free invalidates self.
    auto *type = Py_TYPE(self);
    auto *py_func_rec = reinterpret_cast<function_record_PyObject *>(self);
    cpp_function::destruct(py_func_rec->cpp_func_rec);
    py_func_rec->cpp_func_rec = nullptr;
    // PyObject_New increments the heap type refcount and allocates via
    // PyObject_Malloc; balance both here
    PyObject_Free(self);
    Py_DECREF(type);
}

PYBIND11_NAMESPACE_END(function_record_PyTypeObject_methods)
PYBIND11_INLINE PyObject *get_cached_module(pybind11::str const &nameobj) {
    dict state = detail::get_python_state_dict();
    if (!state.contains("__pybind11_module_cache")) {
        return nullptr;
    }
    dict cache = state["__pybind11_module_cache"];
    if (!cache.contains(nameobj)) {
        return nullptr;
    }
    return cache[nameobj].ptr();
}

PYBIND11_INLINE void cache_completed_module(pybind11::object const &mod) {
    dict state = detail::get_python_state_dict();
    if (!state.contains("__pybind11_module_cache")) {
        state["__pybind11_module_cache"] = dict();
    }
    state["__pybind11_module_cache"][mod.attr("__spec__").attr("name")] = mod;
}

PYBIND11_INLINE PyObject *cached_create_module(PyObject *spec, PyModuleDef *) {
    (void) &cache_completed_module; // silence unused-function warnings, it is used in a macro

    auto nameobj = getattr(reinterpret_borrow<object>(spec), "name", none());
    if (nameobj.is_none()) {
        set_error(PyExc_ImportError, "module spec is missing a name");
        return nullptr;
    }

    auto *mod = get_cached_module(nameobj);
    if (mod) {
        Py_INCREF(mod);
    } else {
        mod = PyModule_NewObject(nameobj.ptr());
    }
    return mod;
}

PYBIND11_NAMESPACE_END(detail)
PYBIND11_INLINE dict globals() {
#if PY_VERSION_HEX >= 0x030d0000
    PyObject *p = PyEval_GetFrameGlobals();
    return p ? reinterpret_steal<dict>(p)
             : reinterpret_borrow<dict>(module_::import("__main__").attr("__dict__").ptr());
#else
    PyObject *p = PyEval_GetGlobals();
    return reinterpret_borrow<dict>(p ? p : module_::import("__main__").attr("__dict__").ptr());
#endif
}

PYBIND11_NAMESPACE_BEGIN(detail)
PYBIND11_INLINE void call_operator_delete(void *p, size_t s, size_t a) {
    (void) s;
    (void) a;
#if defined(__cpp_aligned_new)
    if (a > __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
#    ifdef __cpp_sized_deallocation
        ::operator delete(p, s, std::align_val_t(a));
#    else
        ::operator delete(p, std::align_val_t(a));
#    endif
        return;
    }
#endif
#ifdef __cpp_sized_deallocation
    ::operator delete(p, s);
#else
    ::operator delete(p);
#endif
}

PYBIND11_INLINE void add_class_method(object &cls, const char *name_, const cpp_function &cf) {
    cls.attr(cf.name()) = cf;
    if (std::strcmp(name_, "__eq__") == 0 && !cls.attr("__dict__").contains("__hash__")) {
        cls.attr("__hash__") = none();
    }
}

PYBIND11_INLINE str enum_name(handle arg) {
    dict entries = type::handle_of(arg).attr("__entries");
    for (auto kv : entries) {
        if (handle(kv.second[int_(0)]).equal(arg)) {
            return pybind11::str(kv.first);
        }
    }
    return "???";
}

PYBIND11_NOINLINE_ATTR PYBIND11_INLINE void keep_alive_impl(handle nurse, handle patient) {
    if (!nurse || !patient) {
        pybind11_fail("Could not activate keep_alive!");
    }

    if (patient.is_none() || nurse.is_none()) {
        return; /* Nothing to keep alive or nothing to be kept alive by */
    }

    auto tinfo = all_type_info(Py_TYPE(nurse.ptr()));
    if (!tinfo.empty()) {
        /* It's a pybind-registered type, so we can store the patient in the
         * internal list. */
        add_patient(nurse.ptr(), patient.ptr());
    } else {
        /* Fall back to clever approach based on weak references taken from
         * Boost.Python. This is not used for pybind-registered types because
         * the objects can be destroyed out-of-order in a GC pass. */
        cpp_function disable_lifesupport([patient](handle weakref) {
            patient.dec_ref();
            weakref.dec_ref();
        });

        weakref wr(nurse, disable_lifesupport);

        patient.inc_ref(); /* reference patient and leak the weak reference */
        (void) wr.release();
    }
}

PYBIND11_NOINLINE_ATTR PYBIND11_INLINE void
keep_alive_impl(size_t Nurse, size_t Patient, function_call &call, handle ret) {
    auto get_arg = [&](size_t n) {
        if (n == 0) {
            return ret;
        }
        if (n == 1 && call.init_self) {
            return call.init_self;
        }
        if (n <= call.args.size()) {
            return call.args[n - 1];
        }
        return handle();
    };

    keep_alive_impl(get_arg(Nurse), get_arg(Patient));
}

PYBIND11_INLINE std::pair<decltype(internals::registered_types_py)::iterator, bool>
all_type_info_get_cache(PyTypeObject *type) {
    auto res = with_internals([type](internals &internals) {
        auto ins = internals
                       .registered_types_py
#ifdef __cpp_lib_unordered_map_try_emplace
                       .try_emplace(type);
#else
                       .emplace(type, std::vector<detail::type_info *>());
#endif
        if (ins.second) {
            // For free-threading mode, this call must be under
            // the with_internals() mutex lock, to avoid that other threads
            // continue running with the empty ins.first->second.
            all_type_info_populate(type, ins.first->second);
        }
        return ins;
    });
    if (res.second) {
        // New cache entry created; set up a weak reference to automatically remove it if the type
        // gets destroyed:
        weakref(reinterpret_cast<PyObject *>(type), cpp_function([type](handle wr) {
                    with_internals([type](internals &internals) {
                        internals.registered_types_py.erase(type);

                        // TODO consolidate the erasure code in pybind11_meta_dealloc() in class.h
                        auto &cache = internals.inactive_override_cache;
                        for (auto it = cache.begin(), last = cache.end(); it != last;) {
                            if (it->first == reinterpret_cast<PyObject *>(type)) {
                                it = cache.erase(it);
                            } else {
                                ++it;
                            }
                        }
                    });

                    wr.dec_ref();
                }))
            .release();
    }

    return res;
}

PYBIND11_NAMESPACE_END(detail)
PYBIND11_INLINE void register_exception_translator(ExceptionTranslator &&translator) {
    detail::with_exception_translators(
        [&](std::forward_list<ExceptionTranslator> &exception_translators,
            std::forward_list<ExceptionTranslator> &local_exception_translators) {
            (void) local_exception_translators;
            exception_translators.push_front(std::forward<ExceptionTranslator>(translator));
        });
}

PYBIND11_INLINE void register_local_exception_translator(ExceptionTranslator &&translator) {
    detail::with_exception_translators(
        [&](std::forward_list<ExceptionTranslator> &exception_translators,
            std::forward_list<ExceptionTranslator> &local_exception_translators) {
            (void) exception_translators;
            local_exception_translators.push_front(std::forward<ExceptionTranslator>(translator));
        });
}

PYBIND11_NAMESPACE_BEGIN(detail)
PYBIND11_NOINLINE_ATTR PYBIND11_INLINE void print(const tuple &args, const dict &kwargs) {
#if PY_VERSION_HEX >= 0x030D0000
    auto builtins = reinterpret_steal<dict>(PyEval_GetFrameBuiltins());
#else
    auto builtins = reinterpret_borrow<dict>(PyEval_GetBuiltins());
#endif
    // The builtins dictionary may already be partially cleared during interpreter shutdown.
    auto native_print = reinterpret_steal<object>(dict_getitemstringref(builtins.ptr(), "print"));
    if (!native_print) {
        return;
    }
    auto result
        = reinterpret_steal<object>(PyObject_Call(native_print.ptr(), args.ptr(), kwargs.ptr()));
    if (!result) {
        throw error_already_set();
    }
}

PYBIND11_NAMESPACE_END(detail)
PYBIND11_INLINE void
error_already_set::m_fetched_error_deleter(detail::error_fetch_and_normalize *raw_ptr) {
    gil_scoped_acquire gil;
    error_scope scope;
    delete raw_ptr;
}

PYBIND11_INLINE const char *error_already_set::what() const noexcept {
    gil_scoped_acquire gil;
    error_scope scope;
    return m_fetched_error->error_string().c_str();
}

PYBIND11_NAMESPACE_BEGIN(detail)
PYBIND11_INLINE function
get_type_override(const void *this_ptr, const type_info *this_type, const char *name) {
    handle self = get_object_handle(this_ptr, this_type);
    if (!self) {
        return function();
    }
    handle type = type::handle_of(self);
    auto key = std::make_pair(type.ptr(), name);

    /* Cache functions that aren't overridden in Python to avoid
       many costly Python dictionary lookups below */
    bool not_overridden = with_internals([&key](internals &internals) {
        auto &cache = internals.inactive_override_cache;
        return cache.find(key) != cache.end();
    });
    if (not_overridden) {
        return function();
    }

    function override = getattr(self, name, function());
    if (override.is_cpp_function()) {
        with_internals([&](internals &internals) {
            internals.inactive_override_cache.insert(std::move(key));
        });
        return function();
    }

    /* Don't call dispatch code if invoked from overridden function.
       Unfortunately this doesn't work on PyPy and GraalPy. */
#if !defined(PYPY_VERSION) && !defined(GRAALVM_PYTHON)
    PyFrameObject *frame = PyThreadState_GetFrame(PyThreadState_Get());
    if (frame != nullptr) {
        PyCodeObject *f_code = PyFrame_GetCode(frame);
        // f_code is guaranteed to not be NULL
        if (std::string(str(f_code->co_name)) == name && f_code->co_argcount > 0) {
#    if PY_VERSION_HEX >= 0x030d0000
            PyObject *locals = PyEval_GetFrameLocals();
#    else
            PyObject *locals = PyEval_GetLocals();
            Py_XINCREF(locals);
#    endif
            if (locals != nullptr) {
#    if PY_VERSION_HEX >= 0x030b0000
                PyObject *co_varnames = PyCode_GetVarnames(f_code);
#    else
                PyObject *co_varnames = PyObject_GetAttrString((PyObject *) f_code, "co_varnames");
#    endif
                PyObject *self_arg = PyTuple_GET_ITEM(co_varnames, 0);
                Py_DECREF(co_varnames);
                PyObject *self_caller = dict_getitem(locals, self_arg);
                Py_DECREF(locals);
                if (self_caller == self.ptr()) {
                    Py_DECREF(f_code);
                    Py_DECREF(frame);
                    return function();
                }
            }
        }
        Py_DECREF(f_code);
        Py_DECREF(frame);
    }

#else
    /* PyPy currently doesn't provide a detailed cpyext emulation of
       frame objects, so we have to emulate this using Python. This
       is going to be slow..*/
    dict d;
    d["self"] = self;
    d["name"] = pybind11::str(name);
    PyObject *result
        = PyRun_String("import inspect\n"
                       "frame = inspect.currentframe()\n"
                       "if frame is not None:\n"
                       "    frame = frame.f_back\n"
                       "    if frame is not None and str(frame.f_code.co_name) == name and "
                       "frame.f_code.co_argcount > 0:\n"
                       "        self_caller = frame.f_locals[frame.f_code.co_varnames[0]]\n"
                       "        if self_caller == self:\n"
                       "            self = None\n",
                       Py_file_input,
                       d.ptr(),
                       d.ptr());
    if (result == nullptr)
        throw error_already_set();
    Py_DECREF(result);
    if (d["self"].is_none())
        return function();
#endif

    return override;
}

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

PYBIND11_NOINLINE_ATTR PYBIND11_INLINE cpp_function::unique_function_record
cpp_function::make_function_record() {
    return unique_function_record(new detail::function_record());
}


PYBIND11_INLINE void cpp_function::initialize_generic(unique_function_record &&unique_rec,
                        const char *text,
                        const std::type_info *const *types,
                        size_t args) {
    // Do NOT receive `unique_rec` by value. If this function fails to move out the unique_ptr,
    // we do not want this to destruct the pointer. `initialize` (the caller) still relies on
    // the pointee being alive after this call. Only move out if a `capsule` is going to keep
    // it alive.
    auto *rec = unique_rec.get();

    // Keep track of strdup'ed strings, and clean them up as long as the function's capsule
    // has not taken ownership yet (when `unique_rec.release()` is called).
    // Note: This cannot easily be fixed by a `unique_ptr` with custom deleter, because the
    // strings are only referenced before strdup'ing. So only *after* the following block could
    // `destruct` safely be called, but even then, `repr` could still throw in the middle of
    // copying all strings.
    strdup_guard guarded_strdup;

    /* Create copies of all referenced C-style strings */
    rec->name = guarded_strdup(rec->name ? rec->name : "");
    if (rec->doc) {
        rec->doc = guarded_strdup(rec->doc);
    }
    for (auto &a : rec->args) {
        if (a.name) {
            a.name = guarded_strdup(a.name);
        }
        if (a.descr) {
            a.descr = guarded_strdup(a.descr);
        } else if (a.value) {
            a.descr = guarded_strdup(repr(a.value).cast<std::string>().c_str());
        }
    }

    rec->is_constructor = (std::strcmp(rec->name, "__init__") == 0)
                          || (std::strcmp(rec->name, "__setstate__") == 0);

#if defined(PYBIND11_DETAILED_ERROR_MESSAGES) && !defined(PYBIND11_DISABLE_NEW_STYLE_INIT_WARNING)
    if (rec->is_constructor && !rec->is_new_style_constructor) {
        const auto class_name
            = detail::get_fully_qualified_tp_name((PyTypeObject *) rec->scope.ptr());
        const auto func_name = std::string(rec->name);
        PyErr_WarnEx(PyExc_FutureWarning,
                     ("pybind11-bound class '" + class_name
                      + "' is using an old-style "
                        "placement-new '"
                      + func_name
                      + "' which has been deprecated. See "
                        "the upgrade guide in pybind11's docs. This message is only visible "
                        "when compiled in debug mode.")
                         .c_str(),
                     0);
    }
#endif

    size_t type_index = 0, arg_index = 0;
    std::string signature
        = detail::generate_function_signature(text, rec, types, type_index, arg_index);

    if (arg_index != args - rec->has_args - rec->has_kwargs || types[type_index] != nullptr) {
        pybind11_fail("Internal error while parsing type signature (2)");
    }

    rec->signature = guarded_strdup(signature.c_str());
    rec->args.shrink_to_fit();
    rec->nargs = static_cast<std::uint16_t>(args);

    if (rec->sibling && PYBIND11_INSTANCE_METHOD_CHECK(rec->sibling.ptr())) {
        rec->sibling = PYBIND11_INSTANCE_METHOD_GET_FUNCTION(rec->sibling.ptr());
    }

    detail::function_record *chain = nullptr, *chain_start = rec;
    if (rec->sibling) {
        if (PyCFunction_Check(rec->sibling.ptr())) {
            auto *self = PyCFunction_GET_SELF(rec->sibling.ptr());
            if (self == nullptr) {
                pybind11_fail(
                    "initialize_generic: Unexpected nullptr from PyCFunction_GET_SELF");
            }
            chain = detail::function_record_ptr_from_PyObject(self);
            if (chain && !chain->scope.is(rec->scope)) {
                /* Never append a method to an overload chain of a parent class;
                   instead, hide the parent's overloads in this case */
                chain = nullptr;
            }
        }
        // Don't trigger for things like the default __init__, which are wrapper_descriptors
        // that we are intentionally replacing
        else if (!rec->sibling.is_none() && rec->name[0] != '_') {
            pybind11_fail("Cannot overload existing non-function object \""
                          + std::string(rec->name) + "\" with a function of the same name");
        }
    }

    if (!chain) {
        /* No existing overload was found, create a new function object */
        rec->def = new PyMethodDef();
        std::memset(rec->def, 0, sizeof(PyMethodDef));
        rec->def->ml_name = rec->name;
        rec->def->ml_meth
            = reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)()>(dispatcher));
        rec->def->ml_flags = METH_FASTCALL | METH_KEYWORDS;

        object py_func_rec = detail::function_record_PyObject_New();
        (reinterpret_cast<detail::function_record_PyObject *>(py_func_rec.ptr()))->cpp_func_rec
            = unique_rec.release();
        guarded_strdup.release();

        object scope_module = detail::get_scope_module(rec->scope);
        m_ptr = PyCFunction_NewEx(rec->def, py_func_rec.ptr(), scope_module.ptr());
        if (!m_ptr) {
            pybind11_fail("cpp_function::cpp_function(): Could not allocate function object");
        }
    } else {
        /* Append at the beginning or end of the overload chain */
        m_ptr = rec->sibling.ptr();
        inc_ref();
        if (chain->is_method != rec->is_method) {
            pybind11_fail(
                "overloading a method with both static and instance methods is not supported; "
#if !defined(PYBIND11_DETAILED_ERROR_MESSAGES)
                "#define PYBIND11_DETAILED_ERROR_MESSAGES or compile in debug mode for more "
                "details"
#else
                "error while attempting to bind "
                + std::string(rec->is_method ? "instance" : "static") + " method "
                + std::string(pybind11::str(rec->scope.attr("__name__"))) + "."
                + std::string(rec->name) + signature
#endif
            );
        }

        if (rec->prepend) {
            // Beginning of chain; we need to replace the capsule's current head-of-the-chain
            // pointer with this one, then make this one point to the previous head of the
            // chain.
            chain_start = rec;
            rec->next = chain;
            auto *py_func_rec = reinterpret_cast<detail::function_record_PyObject *>(
                PyCFunction_GET_SELF(m_ptr));
            py_func_rec->cpp_func_rec = unique_rec.release();
            guarded_strdup.release();
        } else {
            // Or end of chain (normal behavior)
            chain_start = chain;
            while (chain->next) {
                chain = chain->next;
            }
            chain->next = unique_rec.release();
            guarded_strdup.release();
        }
    }

    std::string signatures;
    int index = 0;
    /* Create a nice pydoc rec including all signatures and
       docstrings of the functions in the overload chain */
    if (chain && options::show_function_signatures()
        && std::strcmp(rec->name, "_pybind11_conduit_v1_") != 0) {
        // First a generic signature
        signatures += rec->name;
        signatures += "(*args, **kwargs)\n";
        signatures += "Overloaded function.\n\n";
    }
    // Then specific overload signatures
    bool first_user_def = true;
    for (auto *it = chain_start; it != nullptr; it = it->next) {
        if (options::show_function_signatures()
            && std::strcmp(rec->name, "_pybind11_conduit_v1_") != 0) {
            if (index > 0) {
                signatures += '\n';
            }
            if (chain) {
                signatures += std::to_string(++index) + ". ";
            }
            signatures += rec->name;
            signatures += it->signature;
            signatures += '\n';
        }
        if (it->doc && it->doc[0] != '\0' && options::show_user_defined_docstrings()) {
            // If we're appending another docstring, and aren't printing function signatures,
            // we need to append a newline first:
            if (!options::show_function_signatures()) {
                if (first_user_def) {
                    first_user_def = false;
                } else {
                    signatures += '\n';
                }
            }
            if (options::show_function_signatures()) {
                signatures += '\n';
            }
            signatures += it->doc;
            if (options::show_function_signatures()) {
                signatures += '\n';
            }
        }
    }

    auto *func = reinterpret_cast<PyCFunctionObject *>(m_ptr);
    // Install docstring if it's non-empty (when at least one option is enabled)
    auto *doc = signatures.empty() ? nullptr : PYBIND11_COMPAT_STRDUP(signatures.c_str());
    std::free(const_cast<char *>(PYBIND11_PYCFUNCTION_GET_DOC(func)));
    PYBIND11_PYCFUNCTION_SET_DOC(func, doc);

    if (rec->is_method) {
        m_ptr = PYBIND11_INSTANCE_METHOD_NEW(m_ptr, rec->scope.ptr());
        if (!m_ptr) {
            pybind11_fail(
                "cpp_function::cpp_function(): Could not allocate instance method object");
        }
        Py_DECREF(func);
    }
}


PYBIND11_INLINE void cpp_function::destruct(detail::function_record *rec, bool free_strings) {
// If on Python 3.9, check the interpreter "MICRO" (patch) version.
// If this is running on 3.9.0, we have to work around a bug.
#if !defined(PYPY_VERSION) && PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION == 9
    static bool is_zero = Py_GetVersion()[4] == '0';
#endif

    while (rec) {
        detail::function_record *next = rec->next;
        if (rec->free_data) {
            rec->free_data(rec);
        }
        // During initialization, these strings might not have been copied yet,
        // so they cannot be freed. Once the function has been created, they can.
        // Check `make_function_record` for more details.
        if (free_strings) {
            std::free(rec->name);
            std::free(rec->doc);
            std::free(rec->signature);
            for (auto &arg : rec->args) {
                std::free(const_cast<char *>(arg.name));
                std::free(const_cast<char *>(arg.descr));
            }
        }
        for (auto &arg : rec->args) {
            arg.value.dec_ref();
        }
        if (rec->def) {
            std::free(const_cast<char *>(rec->def->ml_doc));
// Python 3.9.0 decref's these in the wrong order; rec->def
// If loaded on 3.9.0, let these leak (use Python 3.9.1 at runtime to fix)
// See https://github.com/python/cpython/pull/22670
#if !defined(PYPY_VERSION) && PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION == 9
            if (!is_zero) {
                delete rec->def;
            }
#else
            delete rec->def;
#endif
        }
        delete rec;
        rec = next;
    }
}


PYBIND11_INLINE PyObject *cpp_function::dispatcher(PyObject *self,
                                                   PyObject *const *args_in_arr,
                                                   size_t nargsf,
                                                   PyObject *kwnames_in) {
    using namespace detail;
    const function_record *overloads = function_record_ptr_from_PyObject(self);
    assert(overloads != nullptr);

    /* Iterator over the list of potentially admissible overloads */
    const function_record *current_overload = overloads;

    /* Need to know how many arguments + keyword arguments there are to pick the right
       overload */
    const auto n_args_in = static_cast<size_t>(PyVectorcall_NARGS(nargsf));

    handle parent = n_args_in > 0 ? args_in_arr[0] : nullptr,
           result = PYBIND11_TRY_NEXT_OVERLOAD;

    auto self_value_and_holder = value_and_holder();
    if (overloads->is_constructor) {
        if (!parent
            || !PyObject_TypeCheck(parent.ptr(), (PyTypeObject *) overloads->scope.ptr())) {
            set_error(PyExc_TypeError,
                      "__init__(self, ...) called with invalid or missing `self` argument");
            return nullptr;
        }

        auto *const tinfo
            = get_type_info(reinterpret_cast<PyTypeObject *>(overloads->scope.ptr()));
        auto *const pi = reinterpret_cast<instance *>(parent.ptr());
        self_value_and_holder = pi->get_value_and_holder(tinfo, true);

        // If this value is already registered it must mean __init__ is invoked multiple times;
        // we really can't support that in C++, so just ignore the second __init__.
        if (self_value_and_holder.instance_registered()) {
            return none().release().ptr();
        }
    }

    try {
        // We do this in two passes: in the first pass, we load arguments with `convert=false`;
        // in the second, we allow conversion (except for arguments with an explicit
        // py::arg().noconvert()).  This lets us prefer calls without conversion, with
        // conversion as a fallback.
        std::vector<function_call> second_pass;

        // However, if there are no overloads, we can just skip the no-convert pass entirely
        const bool overloaded
            = current_overload != nullptr && current_overload->next != nullptr;

        for (; current_overload != nullptr; current_overload = current_overload->next) {

            /* For each overload:
               1. Copy all positional arguments we were given, also checking to make sure that
                  named positional arguments weren't *also* specified via kwarg.
               2. If we weren't given enough, try to make up the omitted ones by checking
                  whether they were provided by a kwarg matching the `py::arg("name")` name. If
                  so, use it (and remove it from kwargs); if not, see if the function binding
                  provided a default that we can use.
               3. Ensure that either all keyword arguments were "consumed", or that the
               function takes a kwargs argument to accept unconsumed kwargs.
               4. Any positional arguments still left get put into a tuple (for args), and any
                  leftover kwargs get put into a dict.
               5. Pack everything into a vector; if we have py::args or py::kwargs, they are an
                  extra tuple or dict at the end of the positional arguments.
               6. Call the function call dispatcher (function_record::impl)

               If one of these fail, move on to the next overload and keep trying until we get
               a result other than PYBIND11_TRY_NEXT_OVERLOAD.
             */

            const function_record &func = *current_overload;
            size_t num_args = func.nargs; // Number of positional arguments that we need
            if (func.has_args) {
                --num_args; // (but don't count py::args
            }
            if (func.has_kwargs) {
                --num_args; //  or py::kwargs)
            }
            size_t pos_args = func.nargs_pos;

            if (!func.has_args && n_args_in > pos_args) {
                continue; // Too many positional arguments for this overload
            }

            if (n_args_in < pos_args && func.args.size() < pos_args) {
                continue; // Not enough positional arguments given, and not enough defaults to
                          // fill in the blanks
            }

            function_call call(func, parent);

            // Protect std::min with parentheses
            size_t args_to_copy = (std::min) (pos_args, n_args_in);
            size_t args_copied = 0;

            // 0. Inject new-style `self` argument
            if (func.is_new_style_constructor) {
                // The `value` may have been preallocated by an old-style `__init__`
                // if it was a preceding candidate for overload resolution.
                if (self_value_and_holder) {
                    self_value_and_holder.type->dealloc(self_value_and_holder);
                }

                call.init_self = args_in_arr[0];
                call.args.emplace_back(reinterpret_cast<PyObject *>(&self_value_and_holder));
                call.args_convert.push_back(false);
                ++args_copied;
            }

            // 1. Copy any position arguments given.
            bool bad_arg = false;
            for (; args_copied < args_to_copy; ++args_copied) {
                const argument_record *arg_rec
                    = args_copied < func.args.size() ? &func.args[args_copied] : nullptr;

                /* if the argument is listed in the call site's kwargs, but the argument is
                also fulfilled positionally, then the call can't match this overload. for
                example, the call site is: foo(0, key=1) but our overload is foo(key:int) then
                this call can't be for us, because it would be invalid.
                */
                if (kwnames_in && arg_rec && arg_rec->name
                    && keyword_index(kwnames_in, arg_rec->name) >= 0) {
                    bad_arg = true;
                    break;
                }

                handle arg(args_in_arr[args_copied]);
                if (arg_rec && !arg_rec->none && arg.is_none()) {
                    bad_arg = true;
                    break;
                }

                call.args.push_back(arg);
                call.args_convert.push_back(arg_rec ? arg_rec->convert : true);
            }
            if (bad_arg) {
                continue; // Maybe it was meant for another overload (issue #688)
            }

            // Keep track of how many position args we copied out in case we need to come back
            // to copy the rest into a py::args argument.
            size_t positional_args_copied = args_copied;

            // 1.5. Fill in any missing pos_only args from defaults if they exist
            if (args_copied < func.nargs_pos_only) {
                for (; args_copied < func.nargs_pos_only; ++args_copied) {
                    const auto &arg_rec = func.args[args_copied];
                    if (arg_rec.value) {
                        call.args.push_back(arg_rec.value);
                        call.args_convert.push_back(arg_rec.convert);
                    } else {
                        break;
                    }
                }

                if (args_copied < func.nargs_pos_only) {
                    continue; // Not enough defaults to fill the positional arguments
                }
            }

            // 2. Check kwargs and, failing that, defaults that may help complete the list
            small_vector<bool, arg_vector_small_size> used_kwargs(
                kwnames_in ? static_cast<size_t>(PyTuple_GET_SIZE(kwnames_in)) : 0, false);
            size_t used_kwargs_count = 0;
            if (args_copied < num_args) {
                for (; args_copied < num_args; ++args_copied) {
                    const auto &arg_rec = func.args[args_copied];

                    handle value;
                    if (kwnames_in && arg_rec.name) {
                        ssize_t i = keyword_index(kwnames_in, arg_rec.name);
                        if (i >= 0) {
                            value = args_in_arr[n_args_in + static_cast<size_t>(i)];
                            used_kwargs.set(static_cast<size_t>(i), true);
                            used_kwargs_count++;
                        }
                    }

                    if (!value) {
                        value = arg_rec.value;
                        if (!value) {
                            break;
                        }
                    }

                    if (!arg_rec.none && value.is_none()) {
                        break;
                    }

                    // If we're at the py::args index then first insert a stub for it to be
                    // replaced later
                    if (func.has_args && call.args.size() == func.nargs_pos) {
                        call.args.push_back(none());
                    }

                    call.args.push_back(value);
                    call.args_convert.push_back(arg_rec.convert);
                }

                if (args_copied < num_args) {
                    continue; // Not enough arguments, defaults, or kwargs to fill the
                              // positional arguments
                }
            }

            // 3. Check everything was consumed (unless we have a kwargs arg)
            if (!func.has_kwargs && used_kwargs_count < used_kwargs.size()) {
                continue; // Unconsumed kwargs, but no py::kwargs argument to accept them
            }

            // 4a. If we have a py::args argument, create a new tuple with leftovers
            if (func.has_args) {
                if (positional_args_copied >= n_args_in) {
                    call.args_ref = tuple(0);
                } else {
                    size_t args_size = n_args_in - positional_args_copied;
                    tuple extra_args(args_size);
                    for (size_t i = 0; i < args_size; ++i) {
                        extra_args[i] = args_in_arr[positional_args_copied + i];
                    }
                    call.args_ref = std::move(extra_args);
                }
                if (call.args.size() <= func.nargs_pos) {
                    call.args.push_back(call.args_ref);
                } else {
                    call.args[func.nargs_pos] = call.args_ref;
                }
                call.args_convert.push_back(false);
            }

            // 4b. If we have a py::kwargs, pass on any remaining kwargs
            if (func.has_kwargs) {
                dict kwargs;
                for (size_t i = 0; i < used_kwargs.size(); ++i) {
                    if (!used_kwargs[i]) {
                        // Cast values into handles before indexing into kwargs to ensure
                        // well-defined evaluation order (MSVC C4866).
                        handle arg_in_arr = args_in_arr[n_args_in + i],
                               kwname = PyTuple_GET_ITEM(kwnames_in, i);
                        kwargs[kwname] = arg_in_arr;
                    }
                }
                call.args.push_back(kwargs);
                call.args_convert.push_back(false);
                call.kwargs_ref = std::move(kwargs);
            }

            // 5. Put everything in a vector.  Not technically step 5, we've been building it
            // in `call.args` all along.

#if defined(PYBIND11_DETAILED_ERROR_MESSAGES)
            if (call.args.size() != func.nargs || call.args_convert.size() != func.nargs) {
                pybind11_fail("Internal error: function call dispatcher inserted wrong number "
                              "of arguments!");
            }
#endif

            args_convert_vector<arg_vector_small_size> second_pass_convert;
            if (overloaded) {
                // We're in the first no-convert pass, so swap out the conversion flags for a
                // set of all-false flags.  If the call fails, we'll swap the flags back in for
                // the conversion-allowed call below.
                second_pass_convert = std::move(call.args_convert);
                call.args_convert
                    = args_convert_vector<arg_vector_small_size>(func.nargs, false);
            }

            // 6. Call the function.
            try {
                loader_life_support guard{};
                result = func.impl(call);
            } catch (reference_cast_error &) {
                result = PYBIND11_TRY_NEXT_OVERLOAD;
            }

            if (result.ptr() != PYBIND11_TRY_NEXT_OVERLOAD) {
                break;
            }

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
            // The no-conversion pass finished without success, try again with conversion
            // allowed
            for (auto &call : second_pass) {
                try {
                    loader_life_support guard{};
                    result = call.func.impl(call);
                } catch (reference_cast_error &) {
                    result = PYBIND11_TRY_NEXT_OVERLOAD;
                }

                if (result.ptr() != PYBIND11_TRY_NEXT_OVERLOAD) {
                    // The error reporting logic below expects 'current_overload' to be valid,
                    // as it would be if we'd encountered this failure in the first-pass loop.
                    if (!result) {
                        current_overload = &call.func;
                    }
                    break;
                }
            }
        }
    } catch (error_already_set &e) {
        e.restore();
        return nullptr;
#ifdef __GLIBCXX__
    } catch (abi::__forced_unwind &) {
        throw;
#endif
    } catch (...) {
        try_translate_exceptions();
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
        if (overloads->is_operator) {
            return handle(Py_NotImplemented).inc_ref().ptr();
        }

        std::string msg = std::string(overloads->name) + "(): incompatible "
                          + std::string(overloads->is_constructor ? "constructor" : "function")
                          + " arguments. The following argument types are supported:\n";

        int ctr = 0;
        for (const function_record *it2 = overloads; it2 != nullptr; it2 = it2->next) {
            msg += "    " + std::to_string(++ctr) + ". ";

            bool wrote_sig = false;
            if (overloads->is_constructor) {
                // For a constructor, rewrite `(self: Object, arg0, ...) -> NoneType` as
                // `Object(arg0, ...)`
                std::string sig = it2->signature;
                size_t start = sig.find('(') + 7; // skip "(self: "
                if (start < sig.size()) {
                    // End at the , for the next argument
                    size_t end = sig.find(", "), next = end + 2;
                    size_t ret = sig.rfind(" -> ");
                    // Or the ), if there is no comma:
                    if (end >= sig.size()) {
                        next = end = sig.find(')');
                    }
                    if (start < end && next < sig.size()) {
                        msg.append(sig, start, end - start);
                        msg += '(';
                        msg.append(sig, next, ret - next);
                        wrote_sig = true;
                    }
                }
            }
            if (!wrote_sig) {
                msg += it2->signature;
            }

            msg += '\n';
        }
        msg += "\nInvoked with: ";
        bool some_args = false;
        for (size_t ti = overloads->is_constructor ? 1 : 0; ti < n_args_in; ++ti) {
            if (!some_args) {
                some_args = true;
            } else {
                msg += ", ";
            }
            try {
                msg += pybind11::repr(args_in_arr[ti]);
            } catch (const error_already_set &) {
                msg += "<repr raised Error>";
            }
        }
        if (kwnames_in && PyTuple_GET_SIZE(kwnames_in) > 0) {
            if (some_args) {
                msg += "; ";
            }
            msg += "kwargs: ";
            bool first = true;
            for (size_t i = 0; i < static_cast<size_t>(PyTuple_GET_SIZE(kwnames_in)); ++i) {
                if (first) {
                    first = false;
                } else {
                    msg += ", ";
                }
                msg += reinterpret_borrow<pybind11::str>(PyTuple_GET_ITEM(kwnames_in, i));
                msg += '=';
                try {
                    msg += pybind11::repr(args_in_arr[n_args_in + i]);
                } catch (const error_already_set &) {
                    msg += "<repr raised Error>";
                }
            }
        }

        append_note_if_missing_header_is_suspected(msg);
        // Attach additional error info to the exception if supported
        if (PyErr_Occurred()) {
            // #HelpAppreciated: unit test coverage for this branch.
            raise_from(PyExc_TypeError, msg.c_str());
            return nullptr;
        }
        set_error(PyExc_TypeError, msg.c_str());
        return nullptr;
    }
    if (!result) {
        std::string msg = "Unable to convert function return value to a "
                          "Python type! The signature was\n\t";
        assert(current_overload != nullptr);
        msg += current_overload->signature;
        append_note_if_missing_header_is_suspected(msg);
        // Attach additional error info to the exception if supported
        if (PyErr_Occurred()) {
            raise_from(PyExc_TypeError, msg.c_str());
            return nullptr;
        }
        set_error(PyExc_TypeError, msg.c_str());
        return nullptr;
    }
    if (overloads->is_constructor && !self_value_and_holder.holder_constructed()) {
        auto *pi = reinterpret_cast<instance *>(parent.ptr());
        self_value_and_holder.type->init_instance(pi, nullptr);
    }
    return result.ptr();
}


PYBIND11_NAMESPACE_BEGIN(detail)

PYBIND11_INLINE void generic_type::initialize(const type_record &rec) {
    if (rec.scope && hasattr(rec.scope, "__dict__")
        && rec.scope.attr("__dict__").contains(rec.name)) {
        pybind11_fail("generic_type: cannot initialize type \"" + std::string(rec.name)
                      + "\": an object with that name is already defined");
    }

    if ((rec.module_local ? get_local_type_info(*rec.type) : get_global_type_info(*rec.type))
        != nullptr) {
        pybind11_fail("generic_type: type \"" + std::string(rec.name)
                      + "\" is already registered!");
    }

    m_ptr = make_new_python_type(rec);

    /* Register supplemental type information in C++ dict */
    auto *tinfo = new detail::type_info();
    tinfo->type = reinterpret_cast<PyTypeObject *>(m_ptr);
    tinfo->cpptype = rec.type;
    tinfo->type_size = rec.type_size;
    tinfo->type_align = rec.type_align;
    tinfo->operator_new = rec.operator_new;
    tinfo->holder_size_in_ptrs = size_in_ptrs(rec.holder_size);
    tinfo->init_instance = rec.init_instance;
    tinfo->dealloc = rec.dealloc;
    tinfo->get_trampoline_self_life_support = rec.get_trampoline_self_life_support;
    tinfo->simple_type = true;
    tinfo->simple_ancestors = true;
    tinfo->module_local = rec.module_local;
    tinfo->holder_enum_v = rec.holder_enum_v;

    with_internals([&](internals &internals) {
        auto tindex = std::type_index(*rec.type);
        tinfo->direct_conversions = &internals.direct_conversions[tindex];
        auto &local_internals = get_local_internals();
        if (rec.module_local) {
            local_internals.registered_types_cpp[rec.type] = tinfo;
        } else {
            internals.registered_types_cpp[tindex] = tinfo;
#if PYBIND11_INTERNALS_VERSION >= 12
            internals.registered_types_cpp_fast[rec.type] = tinfo;
#endif
        }

        PYBIND11_WARNING_PUSH
#if defined(__GNUC__) && __GNUC__ == 12
        // When using GCC 12 these warnings are disabled as they trigger
        // false positive warnings.  Discussed here:
        // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=115824.
        PYBIND11_WARNING_DISABLE_GCC("-Warray-bounds")
        PYBIND11_WARNING_DISABLE_GCC("-Wstringop-overread")
#endif
        internals.registered_types_py[reinterpret_cast<PyTypeObject *>(m_ptr)] = {tinfo};
        PYBIND11_WARNING_POP
    });

    if (rec.bases.size() > 1 || rec.multiple_inheritance) {
        mark_parents_nonsimple(tinfo->type);
        tinfo->simple_ancestors = false;
    } else if (rec.bases.size() == 1) {
        auto *parent_tinfo
            = get_type_info(reinterpret_cast<PyTypeObject *>(rec.bases[0].ptr()));
        assert(parent_tinfo != nullptr);
        bool parent_simple_ancestors = parent_tinfo->simple_ancestors;
        tinfo->simple_ancestors = parent_simple_ancestors;
        // The parent can no longer be a simple type if it has MI and has a child
        parent_tinfo->simple_type = parent_tinfo->simple_type && parent_simple_ancestors;
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
        auto *tinfo2 = get_type_info(reinterpret_cast<PyTypeObject *>(h.ptr()));
        if (tinfo2) {
            tinfo2->simple_type = false;
        }
        mark_parents_nonsimple(reinterpret_cast<PyTypeObject *>(h.ptr()));
    }
}


PYBIND11_NOINLINE_ATTR PYBIND11_INLINE void enum_base::init(bool is_arithmetic, bool is_convertible) {
    m_base.attr("__entries") = dict();
    auto property = handle(reinterpret_cast<PyObject *>(&PyProperty_Type));
    auto static_property
        = handle(reinterpret_cast<PyObject *>(get_internals().static_property_type));

    m_base.attr("__repr__") = cpp_function(
        [](const object &arg) -> str {
            handle type = type::handle_of(arg);
            object type_name = type.attr("__name__");
            return pybind11::str("<{}.{}: {}>")
                .format(std::move(type_name), enum_name(arg), int_(arg));
        },
        name("__repr__"),
        is_method(m_base),
        pos_only());

    m_base.attr("name")
        = property(cpp_function(&enum_name, name("name"), is_method(m_base), pos_only()));

    m_base.attr("__str__") = cpp_function(
        [](handle arg) -> str {
            object type_name = type::handle_of(arg).attr("__name__");
            return pybind11::str("{}.{}").format(std::move(type_name), enum_name(arg));
        },
        name("__str__"),
        is_method(m_base),
        pos_only());

    if (options::show_enum_members_docstring()) {
        m_base.attr("__doc__") = static_property(
            cpp_function(
                [](handle arg) -> std::string {
                    std::string docstring;
                    dict entries = arg.attr("__entries");
                    if ((reinterpret_cast<PyTypeObject *>(arg.ptr()))->tp_doc) {
                        docstring += std::string(
                            reinterpret_cast<PyTypeObject *>(arg.ptr())->tp_doc);
                        docstring += "\n\n";
                    }
                    docstring += "Members:";
                    for (auto kv : entries) {
                        auto key = std::string(pybind11::str(kv.first));
                        auto comment = kv.second[int_(1)];
                        docstring += "\n\n  ";
                        docstring += key;
                        if (!comment.is_none()) {
                            docstring += " : ";
                            docstring += pybind11::str(comment).cast<std::string>();
                        }
                    }
                    return docstring;
                },
                name("__doc__")),
            none(),
            none(),
            "");
    }

    m_base.attr("__members__") = static_property(cpp_function(
                                                     [](handle arg) -> dict {
                                                         dict entries = arg.attr("__entries"),
                                                              m;
                                                         for (auto kv : entries) {
                                                             m[kv.first] = kv.second[int_(0)];
                                                         }
                                                         return m;
                                                     },
                                                     name("__members__")),
                                                 none(),
                                                 none(),
                                                 "");

#define PYBIND11_ENUM_OP_STRICT(op, expr, strict_behavior)                                        \
m_base.attr(op) = cpp_function(                                                               \
    [](const object &a, const object &b) {                                                    \
        if (!type::handle_of(a).is(type::handle_of(b)))                                       \
            strict_behavior; /* NOLINT(bugprone-macro-parentheses) */                         \
        return expr;                                                                          \
    },                                                                                        \
    name(op),                                                                                 \
    is_method(m_base),                                                                        \
    arg("other"),                                                                             \
    pos_only())

#define PYBIND11_ENUM_OP_CONV(op, expr)                                                           \
m_base.attr(op) = cpp_function(                                                               \
    [](const object &a_, const object &b_) {                                                  \
        int_ a(a_), b(b_);                                                                    \
        return expr;                                                                          \
    },                                                                                        \
    name(op),                                                                                 \
    is_method(m_base),                                                                        \
    arg("other"),                                                                             \
    pos_only())

#define PYBIND11_ENUM_OP_CONV_LHS(op, expr)                                                       \
m_base.attr(op) = cpp_function(                                                               \
    [](const object &a_, const object &b) {                                                   \
        int_ a(a_);                                                                           \
        return expr;                                                                          \
    },                                                                                        \
    name(op),                                                                                 \
    is_method(m_base),                                                                        \
    arg("other"),                                                                             \
    pos_only())

    if (is_convertible) {
        if (is_arithmetic) {
            m_base.attr("__invert__")
                = cpp_function([](const object &arg) { return ~(int_(arg)); },
                               name("__invert__"),
                               is_method(m_base),
                               pos_only());
        }
    }

#undef PYBIND11_ENUM_OP_CONV_LHS
#undef PYBIND11_ENUM_OP_CONV
#undef PYBIND11_ENUM_OP_STRICT

    m_base.attr("__getstate__") = cpp_function([](const object &arg) { return int_(arg); },
                                               name("__getstate__"),
                                               is_method(m_base),
                                               pos_only());

    m_base.attr("__hash__") = cpp_function([](const object &arg) { return int_(arg); },
                                           name("__hash__"),
                                           is_method(m_base),
                                           pos_only());
}


PYBIND11_NOINLINE_ATTR PYBIND11_INLINE void enum_base::value(char const *name_, object value, const char *doc) {
    dict entries = m_base.attr("__entries");
    str name(name_);
    if (entries.contains(name)) {
        std::string type_name = std::string(str(m_base.attr("__name__")));
        throw value_error(std::move(type_name) + ": element \"" + std::string(name_)
                          + "\" already exists!");
    }

    entries[name] = pybind11::make_tuple(value, doc);
    m_base.attr(std::move(name)) = std::move(value);
}


PYBIND11_NOINLINE_ATTR PYBIND11_INLINE void enum_base::export_values() {
    dict entries = m_base.attr("__entries");
    for (auto kv : entries) {
        m_parent.attr(kv.first) = kv.second[int_(0)];
    }
}


PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
