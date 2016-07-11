/*
    pybind11/pybind11.h: Main header file of the C++11 python
    binding generator library

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#  pragma warning(disable: 4800) // warning C4800: 'int': forcing value to bool 'true' or 'false' (performance warning)
#  pragma warning(disable: 4996) // warning C4996: The POSIX name for this item is deprecated. Instead, use the ISO C and C++ conformant name
#  pragma warning(disable: 4100) // warning C4100: Unreferenced formal parameter
#  pragma warning(disable: 4512) // warning C4512: Assignment operator was implicitly defined as deleted
#elif defined(__ICC) || defined(__INTEL_COMPILER)
#  pragma warning(push)
#  pragma warning(disable:2196)  // warning #2196: routine is both "inline" and "noinline"
#elif defined(__GNUG__) && !defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wunused-but-set-parameter"
#  pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#  pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#  pragma GCC diagnostic ignored "-Wstrict-aliasing"
#  pragma GCC diagnostic ignored "-Wattributes"
#endif

#include "attr.h"

NAMESPACE_BEGIN(pybind11)

/// Wraps an arbitrary C++ function/method/lambda function/.. into a callable Python object
class cpp_function : public function {
public:
    cpp_function() { }

    /// Construct a cpp_function from a vanilla function pointer
    template <typename Return, typename... Args, typename... Extra>
    cpp_function(Return (*f)(Args...), const Extra&... extra) {
        initialize(f, f, extra...);
    }

    /// Construct a cpp_function from a lambda function (possibly with internal state)
    template <typename Func, typename... Extra> cpp_function(Func &&f, const Extra&... extra) {
        initialize(std::forward<Func>(f),
                   (typename detail::remove_class<decltype(
                       &std::remove_reference<Func>::type::operator())>::type *) nullptr, extra...);
    }

    /// Construct a cpp_function from a class method (non-const)
    template <typename Return, typename Class, typename... Arg, typename... Extra>
            cpp_function(Return (Class::*f)(Arg...), const Extra&... extra) {
        initialize([f](Class *c, Arg... args) -> Return { return (c->*f)(args...); },
                   (Return (*) (Class *, Arg...)) nullptr, extra...);
    }

    /// Construct a cpp_function from a class method (const)
    template <typename Return, typename Class, typename... Arg, typename... Extra>
            cpp_function(Return (Class::*f)(Arg...) const, const Extra&... extra) {
        initialize([f](const Class *c, Arg... args) -> Return { return (c->*f)(args...); },
                   (Return (*)(const Class *, Arg ...)) nullptr, extra...);
    }

    /// Return the function name
    object name() const { return attr("__name__"); }

protected:
    /// Special internal constructor for functors, lambda functions, etc.
    template <typename Func, typename Return, typename... Args, typename... Extra>
    void initialize(Func &&f, Return (*)(Args...), const Extra&... extra) {
        static_assert(detail::expected_num_args<Extra...>(sizeof...(Args)),
                      "The number of named arguments does not match the function signature");

        struct capture { typename std::remove_reference<Func>::type f; };

        /* Store the function including any extra state it might have (e.g. a lambda capture object) */
        auto rec = new detail::function_record();

        /* Store the capture object directly in the function record if there is enough space */
        if (sizeof(capture) <= sizeof(rec->data)) {
            new ((capture *) &rec->data) capture { std::forward<Func>(f) };
            if (!std::is_trivially_destructible<Func>::value)
                rec->free_data = [](detail::function_record *r) { ((capture *) &r->data)->~capture(); };
        } else {
            rec->data[0] = new capture { std::forward<Func>(f) };
            rec->free_data = [](detail::function_record *r) { delete ((capture *) r->data[0]); };
        }

        /* Type casters for the function arguments and return value */
        typedef detail::type_caster<typename std::tuple<Args...>> cast_in;
        typedef detail::type_caster<typename std::conditional<
            std::is_void<Return>::value, detail::void_type,
            typename detail::intrinsic_type<Return>::type>::type> cast_out;

        /* Dispatch code which converts function arguments and performs the actual function call */
        rec->impl = [](detail::function_record *rec, handle args, handle kwargs, handle parent) -> handle {
            cast_in args_converter;

            /* Try to cast the function arguments into the C++ domain */
            if (!args_converter.load_args(args, kwargs, true))
                return PYBIND11_TRY_NEXT_OVERLOAD;

            /* Invoke call policy pre-call hook */
            detail::process_attributes<Extra...>::precall(args);

            /* Get a pointer to the capture object */
            capture *cap = (capture *) (sizeof(capture) <= sizeof(rec->data)
                                        ? &rec->data : rec->data[0]);

            /* Perform the functioncall */
            handle result = cast_out::cast(args_converter.template call<Return>(cap->f),
                                           rec->policy, parent);

            /* Invoke call policy post-call hook */
            detail::process_attributes<Extra...>::postcall(args, result);

            return result;
        };

        /* Process any user-provided function attributes */
        detail::process_attributes<Extra...>::init(extra..., rec);

        /* Generate a readable signature describing the function's arguments and return value types */
        using detail::descr;
        PYBIND11_DESCR signature = cast_in::name() + detail::_(" -> ") + cast_out::name();

        /* Register the function with Python from generic (non-templated) code */
        initialize_generic(rec, signature.text(), signature.types(), sizeof...(Args));

        if (cast_in::has_args) rec->has_args = true;
        if (cast_in::has_kwargs) rec->has_kwargs = true;
    }

    /// Register a function call with Python (generic non-templated code goes here)
    void initialize_generic(detail::function_record *rec, const char *text,
                            const std::type_info *const *types, int args) {

        /* Create copies of all referenced C-style strings */
        rec->name = strdup(rec->name ? rec->name : "");
        if (rec->doc) rec->doc = strdup(rec->doc);
        for (auto &a: rec->args) {
            if (a.name)
                a.name = strdup(a.name);
            if (a.descr)
                a.descr = strdup(a.descr);
            else if (a.value)
                a.descr = strdup(((std::string) ((object) handle(a.value).attr("__repr__"))().str()).c_str());
        }
        auto const &registered_types = detail::get_internals().registered_types_cpp;

        /* Generate a proper function signature */
        std::string signature;
        size_t type_depth = 0, char_index = 0, type_index = 0, arg_index = 0;
        while (true) {
            char c = text[char_index++];
            if (c == '\0')
                break;

            if (c == '{') {
                if (type_depth == 1 && arg_index < rec->args.size()) {
                    signature += rec->args[arg_index].name;
                    signature += " : ";
                }
                ++type_depth;
            } else if (c == '}') {
                --type_depth;
                if (type_depth == 1 && arg_index < rec->args.size()) {
                    if (rec->args[arg_index].descr) {
                        signature += " = ";
                        signature += rec->args[arg_index].descr;
                    }
                    arg_index++;
                }
            } else if (c == '%') {
                const std::type_info *t = types[type_index++];
                if (!t)
                    pybind11_fail("Internal error while parsing type signature (1)");
                auto it = registered_types.find(std::type_index(*t));
                if (it != registered_types.end()) {
                    signature += ((const detail::type_info *) it->second)->type->tp_name;
                } else {
                    std::string tname(t->name());
                    detail::clean_type_id(tname);
                    signature += tname;
                }
            } else {
                signature += c;
            }
        }
        if (type_depth != 0 || types[type_index] != nullptr)
            pybind11_fail("Internal error while parsing type signature (2)");

        #if !defined(PYBIND11_CPP14)
            delete[] types;
            delete[] text;
        #endif

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
        rec->is_constructor = !strcmp(rec->name, "__init__") || !strcmp(rec->name, "__setstate__");
        rec->has_args = false;
        rec->has_kwargs = false;
        rec->nargs = (uint16_t) args;

#if PY_MAJOR_VERSION < 3
        if (rec->sibling && PyMethod_Check(rec->sibling.ptr()))
            rec->sibling = PyMethod_GET_FUNCTION(rec->sibling.ptr());
#endif

        detail::function_record *chain = nullptr, *chain_start = rec;
        if (rec->sibling && PyCFunction_Check(rec->sibling.ptr())) {
            capsule rec_capsule(PyCFunction_GetSelf(rec->sibling.ptr()), true);
            chain = (detail::function_record *) rec_capsule;
            /* Never append a method to an overload chain of a parent class;
               instead, hide the parent's overloads in this case */
            if (chain->class_ != rec->class_)
                chain = nullptr;
        }

        if (!chain) {
            /* No existing overload was found, create a new function object */
            rec->def = new PyMethodDef();
            memset(rec->def, 0, sizeof(PyMethodDef));
            rec->def->ml_name = rec->name;
            rec->def->ml_meth = reinterpret_cast<PyCFunction>(*dispatcher);
            rec->def->ml_flags = METH_VARARGS | METH_KEYWORDS;

            capsule rec_capsule(rec, [](PyObject *o) {
                destruct((detail::function_record *) PyCapsule_GetPointer(o, nullptr));
            });

            object scope_module;
            if (rec->scope) {
                scope_module = (object) rec->scope.attr("__module__");
                if (!scope_module)
                    scope_module = (object) rec->scope.attr("__name__");
            }

            m_ptr = PyCFunction_NewEx(rec->def, rec_capsule.ptr(), scope_module.ptr());
            if (!m_ptr)
                pybind11_fail("cpp_function::cpp_function(): Could not allocate function object");
        } else {
            /* Append at the end of the overload chain */
            m_ptr = rec->sibling.ptr();
            inc_ref();
            chain_start = chain;
            while (chain->next)
                chain = chain->next;
            chain->next = rec;
        }

        std::string signatures;
        int index = 0;
        /* Create a nice pydoc rec including all signatures and
           docstrings of the functions in the overload chain */
        if (chain) {
            // First a generic signature
            signatures += rec->name;
            signatures += "(*args, **kwargs)\n";
            signatures += "Overloaded function.\n\n";
        }
        // Then specific overload signatures
        for (auto it = chain_start; it != nullptr; it = it->next) {
            if (chain)
                signatures += std::to_string(++index) + ". ";
            signatures += rec->name;
            signatures += it->signature;
            signatures += "\n";
            if (it->doc && strlen(it->doc) > 0) {
                signatures += "\n";
                signatures += it->doc;
                signatures += "\n";
            }
            if (it->next)
                signatures += "\n";
        }

        /* Install docstring */
        PyCFunctionObject *func = (PyCFunctionObject *) m_ptr;
        if (func->m_ml->ml_doc)
            std::free((char *) func->m_ml->ml_doc);
        func->m_ml->ml_doc = strdup(signatures.c_str());

        if (rec->class_) {
            m_ptr = PYBIND11_INSTANCE_METHOD_NEW(m_ptr, rec->class_.ptr());
            if (!m_ptr)
                pybind11_fail("cpp_function::cpp_function(): Could not allocate instance method object");
            Py_DECREF(func);
        }
    }

    /// When a cpp_function is GCed, release any memory allocated by pybind11
    static void destruct(detail::function_record *rec) {
        while (rec) {
            detail::function_record *next = rec->next;
            if (rec->free_data)
                rec->free_data(rec);
            std::free((char *) rec->name);
            std::free((char *) rec->doc);
            std::free((char *) rec->signature);
            for (auto &arg: rec->args) {
                std::free((char *) arg.name);
                std::free((char *) arg.descr);
                arg.value.dec_ref();
            }
            if (rec->def) {
                std::free((char *) rec->def->ml_doc);
                delete rec->def;
            }
            delete rec;
            rec = next;
        }
    }

    /// Main dispatch logic for calls to functions bound using pybind11
    static PyObject *dispatcher(PyObject *self, PyObject *args, PyObject *kwargs) {
        /* Iterator over the list of potentially admissible overloads */
        detail::function_record *overloads = (detail::function_record *) PyCapsule_GetPointer(self, nullptr),
                                *it = overloads;

        /* Need to know how many arguments + keyword arguments there are to pick the right overload */
        size_t nargs = (size_t) PyTuple_GET_SIZE(args),
               nkwargs = kwargs ? (size_t) PyDict_Size(kwargs) : 0;

        handle parent = nargs > 0 ? PyTuple_GET_ITEM(args, 0) : nullptr,
               result = PYBIND11_TRY_NEXT_OVERLOAD;
        try {
            for (; it != nullptr; it = it->next) {
                tuple args_(args, true);
                size_t kwargs_consumed = 0;

                /* For each overload:
                   1. If the required list of arguments is longer than the
                      actually provided amount, create a copy of the argument
                      list and fill in any available keyword/default arguments.
                   2. Ensure that all keyword arguments were "consumed"
                   3. Call the function call dispatcher (function_record::impl)
                 */
                size_t nargs_ = nargs;
                if (nargs < it->args.size()) {
                    nargs_ = it->args.size();
                    args_ = tuple(nargs_);
                    for (size_t i = 0; i < nargs; ++i) {
                        handle item = PyTuple_GET_ITEM(args, i);
                        PyTuple_SET_ITEM(args_.ptr(), i, item.inc_ref().ptr());
                    }

                    int arg_ctr = 0;
                    for (auto const &it2 : it->args) {
                        int index = arg_ctr++;
                        if (PyTuple_GET_ITEM(args_.ptr(), index))
                            continue;

                        handle value;
                        if (kwargs)
                            value = PyDict_GetItemString(kwargs, it2.name);

                        if (value)
                            kwargs_consumed++;
                        else if (it2.value)
                            value = it2.value;

                        if (value) {
                            PyTuple_SET_ITEM(args_.ptr(), index, value.inc_ref().ptr());
                        } else {
                            kwargs_consumed = (size_t) -1; /* definite failure */
                            break;
                        }
                    }
                }

                try {
                    if ((kwargs_consumed == nkwargs || it->has_kwargs) &&
                        (nargs_ == it->nargs || it->has_args))
                        result = it->impl(it, args_, kwargs, parent);
                } catch (cast_error &) {
                    result = PYBIND11_TRY_NEXT_OVERLOAD;
                }

                if (result.ptr() != PYBIND11_TRY_NEXT_OVERLOAD)
                    break;
            }
        } catch (const error_already_set &)      {                                                 return nullptr;
        } catch (const index_error &e)           { PyErr_SetString(PyExc_IndexError,    e.what()); return nullptr;
        } catch (const value_error &e)           { PyErr_SetString(PyExc_ValueError,    e.what()); return nullptr;
        } catch (const stop_iteration &e)        { PyErr_SetString(PyExc_StopIteration, e.what()); return nullptr;
        } catch (const std::bad_alloc &e)        { PyErr_SetString(PyExc_MemoryError,   e.what()); return nullptr;
        } catch (const std::domain_error &e)     { PyErr_SetString(PyExc_ValueError,    e.what()); return nullptr;
        } catch (const std::invalid_argument &e) { PyErr_SetString(PyExc_ValueError,    e.what()); return nullptr;
        } catch (const std::length_error &e)     { PyErr_SetString(PyExc_ValueError,    e.what()); return nullptr;
        } catch (const std::out_of_range &e)     { PyErr_SetString(PyExc_IndexError,    e.what()); return nullptr;
        } catch (const std::range_error &e)      { PyErr_SetString(PyExc_ValueError,    e.what()); return nullptr;
        } catch (const std::exception &e)        { PyErr_SetString(PyExc_RuntimeError,  e.what()); return nullptr;
        } catch (...) {
            PyErr_SetString(PyExc_RuntimeError, "Caught an unknown exception!");
            return nullptr;
        }

        if (result.ptr() == PYBIND11_TRY_NEXT_OVERLOAD) {
            std::string msg = "Incompatible function arguments. The "
                              "following argument types are supported:\n";
            int ctr = 0;
            for (detail::function_record *it2 = overloads; it2 != nullptr; it2 = it2->next) {
                msg += "    "+ std::to_string(++ctr) + ". ";
                msg += it2->signature;
                msg += "\n";
            }
            msg += "    Invoked with: ";
            tuple args_(args, true);
            for( std::size_t ti = 0; ti != args_.size(); ++ti)
            {
                msg += static_cast<std::string>(static_cast<object>(args_[ti]).str());
                if ((ti + 1) != args_.size() )
                    msg += ", ";
            }
            PyErr_SetString(PyExc_TypeError, msg.c_str());
            return nullptr;
        } else if (!result) {
            std::string msg = "Unable to convert function return value to a "
                              "Python type! The signature was\n\t";
            msg += it->signature;
            PyErr_SetString(PyExc_TypeError, msg.c_str());
            return nullptr;
        } else {
            if (overloads->is_constructor) {
                /* When a constructor ran successfully, the corresponding
                   holder type (e.g. std::unique_ptr) must still be initialized. */
                PyObject *inst = PyTuple_GET_ITEM(args, 0);
                auto tinfo = detail::get_type_info(Py_TYPE(inst));
                tinfo->init_holder(inst, nullptr);
            }
            return result.ptr();
        }
    }
};

/// Wrapper for Python extension modules
class module : public object {
public:
    PYBIND11_OBJECT_DEFAULT(module, object, PyModule_Check)

    module(const char *name, const char *doc = nullptr) {
#if PY_MAJOR_VERSION >= 3
        PyModuleDef *def = new PyModuleDef();
        memset(def, 0, sizeof(PyModuleDef));
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

    template <typename Func, typename... Extra>
    module &def(const char *name_, Func &&f, const Extra& ... extra) {
        cpp_function func(std::forward<Func>(f), name(name_),
                          sibling((handle) attr(name_)), scope(*this), extra...);
        /* PyModule_AddObject steals a reference to 'func' */
        PyModule_AddObject(ptr(), name_, func.inc_ref().ptr());
        return *this;
    }

    module def_submodule(const char *name, const char *doc = nullptr) {
        std::string full_name = std::string(PyModule_GetName(m_ptr))
            + std::string(".") + std::string(name);
        module result(PyImport_AddModule(full_name.c_str()), true);
        if (doc)
            result.attr("__doc__") = pybind11::str(doc);
        attr(name) = result;
        return result;
    }

    static module import(const char *name) {
        PyObject *obj = PyImport_ImportModule(name);
        if (!obj)
            pybind11_fail("Module \"" + std::string(name) + "\" not found!");
        return module(obj, false);
    }
};

NAMESPACE_BEGIN(detail)
/// Generic support for creating new Python heap types
class generic_type : public object {
    template <typename type, typename holder_type> friend class class_;
public:
    PYBIND11_OBJECT_DEFAULT(generic_type, object, PyType_Check)
protected:
    void initialize(type_record *rec) {
        if (rec->base_type) {
            if (rec->base_handle)
                pybind11_fail("generic_type: specified base type multiple times!");
            rec->base_handle = detail::get_type_handle(*(rec->base_type));
            if (!rec->base_handle) {
                std::string tname(rec->base_type->name());
                detail::clean_type_id(tname);
                pybind11_fail("generic_type: type \"" + std::string(rec->name) +
                              "\" referenced unknown base type \"" + tname + "\"");
            }
        }

        auto &internals = get_internals();
        auto tindex = std::type_index(*(rec->type));

        if (internals.registered_types_cpp.find(tindex) !=
            internals.registered_types_cpp.end())
            pybind11_fail("generic_type: type \"" + std::string(rec->name) +
                          "\" is already registered!");

        object name(PYBIND11_FROM_STRING(rec->name), false);
        object scope_module;
        if (rec->scope) {
            scope_module = (object) rec->scope.attr("__module__");
            if (!scope_module)
                scope_module = (object) rec->scope.attr("__name__");
        }

#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 3
        /* Qualified names for Python >= 3.3 */
        object scope_qualname;
        if (rec->scope)
            scope_qualname = (object) rec->scope.attr("__qualname__");
        object ht_qualname;
        if (scope_qualname) {
            ht_qualname = object(PyUnicode_FromFormat(
                "%U.%U", scope_qualname.ptr(), name.ptr()), false);
        } else {
            ht_qualname = name;
        }
#endif
        std::string full_name = (scope_module ? ((std::string) scope_module.str() + "." + rec->name)
                                              : std::string(rec->name));

        char *tp_doc = nullptr;
        if (rec->doc) {
            /* Allocate memory for docstring (using PyObject_MALLOC, since
               Python will free this later on) */
            size_t size = strlen(rec->doc) + 1;
            tp_doc = (char *) PyObject_MALLOC(size);
            memcpy((void *) tp_doc, rec->doc, size);
        }

        object type_holder(PyType_Type.tp_alloc(&PyType_Type, 0), false);
        auto type = (PyHeapTypeObject*) type_holder.ptr();

        if (!type_holder || !name)
            pybind11_fail("generic_type: unable to create type object!");

        /* Register supplemental type information in C++ dict */
        detail::type_info *tinfo = new detail::type_info();
        tinfo->type = (PyTypeObject *) type;
        tinfo->type_size = rec->type_size;
        tinfo->init_holder = rec->init_holder;
        internals.registered_types_cpp[tindex] = tinfo;
        internals.registered_types_py[type] = tinfo;

        /* Basic type attributes */
        type->ht_type.tp_name = strdup(full_name.c_str());
        type->ht_type.tp_basicsize = (ssize_t) rec->instance_size;
        type->ht_type.tp_base = (PyTypeObject *) rec->base_handle.ptr();
        rec->base_handle.inc_ref();

        type->ht_name = name.release().ptr();

#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 3
        type->ht_qualname = ht_qualname.release().ptr();
#endif

        /* Supported protocols */
        type->ht_type.tp_as_number = &type->as_number;
        type->ht_type.tp_as_sequence = &type->as_sequence;
        type->ht_type.tp_as_mapping = &type->as_mapping;

        /* Supported elementary operations */
        type->ht_type.tp_init = (initproc) init;
        type->ht_type.tp_new = (newfunc) new_instance;
        type->ht_type.tp_dealloc = rec->dealloc;

        /* Support weak references (needed for the keep_alive feature) */
        type->ht_type.tp_weaklistoffset = offsetof(instance_essentials<void>, weakrefs);

        /* Flags */
        type->ht_type.tp_flags |= Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;
#if PY_MAJOR_VERSION < 3
        type->ht_type.tp_flags |= Py_TPFLAGS_CHECKTYPES;
#endif
        type->ht_type.tp_flags &= ~Py_TPFLAGS_HAVE_GC;

        type->ht_type.tp_doc = tp_doc;

        if (PyType_Ready(&type->ht_type) < 0)
            pybind11_fail("generic_type: PyType_Ready failed!");

        m_ptr = type_holder.ptr();

        if (scope_module) // Needed by pydoc
            attr("__module__") = scope_module;

        /* Register type with the parent scope */
        if (rec->scope)
            rec->scope.attr(handle(type->ht_name)) = *this;

        type_holder.release();
    }

    /// Allocate a metaclass on demand (for static properties)
    handle metaclass() {
        auto &ht_type = ((PyHeapTypeObject *) m_ptr)->ht_type;
        auto &ob_type = PYBIND11_OB_TYPE(ht_type);

        if (ob_type == &PyType_Type) {
            std::string name_ = std::string(ht_type.tp_name) + "__Meta";
#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 3
            object ht_qualname(PyUnicode_FromFormat(
                "%U__Meta", ((object) attr("__qualname__")).ptr()), false);
#endif
            object name(PYBIND11_FROM_STRING(name_.c_str()), false);
            object type_holder(PyType_Type.tp_alloc(&PyType_Type, 0), false);
            if (!type_holder || !name)
                pybind11_fail("generic_type::metaclass(): unable to create type object!");

            auto type = (PyHeapTypeObject*) type_holder.ptr();
            type->ht_name = name.release().ptr();

#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 3
            /* Qualified names for Python >= 3.3 */
            type->ht_qualname = ht_qualname.release().ptr();
#endif
            type->ht_type.tp_name = strdup(name_.c_str());
            type->ht_type.tp_base = ob_type;
            type->ht_type.tp_flags |= (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE) &
                                      ~Py_TPFLAGS_HAVE_GC;

            if (PyType_Ready(&type->ht_type) < 0)
                pybind11_fail("generic_type::metaclass(): PyType_Ready failed!");

            ob_type = (PyTypeObject *) type_holder.release().ptr();
        }
        return handle((PyObject *) ob_type);
    }

    static int init(void *self, PyObject *, PyObject *) {
        std::string msg = std::string(Py_TYPE(self)->tp_name) + ": No constructor defined!";
        PyErr_SetString(PyExc_TypeError, msg.c_str());
        return -1;
    }

    static PyObject *new_instance(PyTypeObject *type, PyObject *, PyObject *) {
        instance<void> *self = (instance<void> *) PyType_GenericAlloc((PyTypeObject *) type, 0);
        auto tinfo = detail::get_type_info(type);
        self->value = ::operator new(tinfo->type_size);
        self->owned = true;
        self->parent = nullptr;
        self->constructed = false;
        detail::get_internals().registered_instances[self->value] = (PyObject *) self;
        return (PyObject *) self;
    }

    static void dealloc(instance<void> *self) {
        if (self->value) {
            bool dont_cache = self->parent && ((instance<void> *) self->parent)->value == self->value;
            if (!dont_cache) { // avoid an issue with internal references matching their parent's address
                auto &registered_instances = detail::get_internals().registered_instances;
                auto it = registered_instances.find(self->value);
                if (it == registered_instances.end())
                    pybind11_fail("generic_type::dealloc(): Tried to deallocate unregistered instance!");
                registered_instances.erase(it);
            }
            Py_XDECREF(self->parent);
            if (self->weakrefs)
                PyObject_ClearWeakRefs((PyObject *) self);
        }
        Py_TYPE(self)->tp_free((PyObject*) self);
    }

    void install_buffer_funcs(
            buffer_info *(*get_buffer)(PyObject *, void *),
            void *get_buffer_data) {
        PyHeapTypeObject *type = (PyHeapTypeObject*) m_ptr;
        type->ht_type.tp_as_buffer = &type->as_buffer;
#if PY_MAJOR_VERSION < 3
        type->ht_type.tp_flags |= Py_TPFLAGS_HAVE_NEWBUFFER;
#endif
        type->as_buffer.bf_getbuffer = getbuffer;
        type->as_buffer.bf_releasebuffer = releasebuffer;
        auto tinfo = detail::get_type_info(&type->ht_type);
        tinfo->get_buffer = get_buffer;
        tinfo->get_buffer_data = get_buffer_data;
    }

    static int getbuffer(PyObject *obj, Py_buffer *view, int flags) {
        auto tinfo = detail::get_type_info(Py_TYPE(obj));
        if (view == nullptr || obj == nullptr || !tinfo || !tinfo->get_buffer) {
            PyErr_SetString(PyExc_BufferError, "generic_type::getbuffer(): Internal error");
            return -1;
        }
        memset(view, 0, sizeof(Py_buffer));
        buffer_info *info = tinfo->get_buffer(obj, tinfo->get_buffer_data);
        view->obj = obj;
        view->ndim = 1;
        view->internal = info;
        view->buf = info->ptr;
        view->itemsize = (ssize_t) info->itemsize;
        view->len = view->itemsize;
        for (auto s : info->shape)
            view->len *= s;
        if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT)
            view->format = const_cast<char *>(info->format.c_str());
        if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
            view->ndim = (int) info->ndim;
            view->strides = (ssize_t *) &info->strides[0];
            view->shape = (ssize_t *) &info->shape[0];
        }
        Py_INCREF(view->obj);
        return 0;
    }

    static void releasebuffer(PyObject *, Py_buffer *view) { delete (buffer_info *) view->internal; }
};
NAMESPACE_END(detail)

template <typename type, typename holder_type = std::unique_ptr<type>>
class class_ : public detail::generic_type {
public:
    typedef detail::instance<type, holder_type> instance_type;

    PYBIND11_OBJECT(class_, detail::generic_type, PyType_Check)

    template <typename... Extra>
    class_(handle scope, const char *name, const Extra &... extra) {
        detail::type_record record;
        record.scope = scope;
        record.name = name;
        record.type = &typeid(type);
        record.type_size = sizeof(type);
        record.instance_size = sizeof(instance_type);
        record.init_holder = init_holder;
        record.dealloc = dealloc;

        /* Process optional arguments, if any */
        detail::process_attributes<Extra...>::init(extra..., &record);

        detail::generic_type::initialize(&record);
    }

    template <typename Func, typename... Extra>
    class_ &def(const char *name_, Func&& f, const Extra&... extra) {
        cpp_function cf(std::forward<Func>(f), name(name_),
                        sibling(attr(name_)), is_method(*this),
                        extra...);
        attr(cf.name()) = cf;
        return *this;
    }

    template <typename Func, typename... Extra> class_ &
    def_static(const char *name_, Func f, const Extra&... extra) {
        cpp_function cf(std::forward<Func>(f), name(name_),
                        sibling(attr(name_)), scope(*this), extra...);
        attr(cf.name()) = cf;
        return *this;
    }

    template <detail::op_id id, detail::op_type ot, typename L, typename R, typename... Extra>
    class_ &def(const detail::op_<id, ot, L, R> &op, const Extra&... extra) {
        op.template execute<type>(*this, extra...);
        return *this;
    }

    template <detail::op_id id, detail::op_type ot, typename L, typename R, typename... Extra>
    class_ & def_cast(const detail::op_<id, ot, L, R> &op, const Extra&... extra) {
        op.template execute_cast<type>(*this, extra...);
        return *this;
    }

    template <typename... Args, typename... Extra>
    class_ &def(const detail::init<Args...> &init, const Extra&... extra) {
        init.template execute<type>(*this, extra...);
        return *this;
    }

    template <typename Func> class_& def_buffer(Func &&func) {
        struct capture { Func func; };
        capture *ptr = new capture { std::forward<Func>(func) };
        install_buffer_funcs([](PyObject *obj, void *ptr) -> buffer_info* {
            detail::type_caster<type> caster;
            if (!caster.load(obj, false))
                return nullptr;
            return new buffer_info(((capture *) ptr)->func(caster));
        }, ptr);
        return *this;
    }

    template <typename C, typename D, typename... Extra>
    class_ &def_readwrite(const char *name, D C::*pm, const Extra&... extra) {
        cpp_function fget([pm](const C &c) -> const D &{ return c.*pm; }, is_method(*this)),
                     fset([pm](C &c, const D &value) { c.*pm = value; }, is_method(*this));
        def_property(name, fget, fset, return_value_policy::reference_internal, extra...);
        return *this;
    }

    template <typename C, typename D, typename... Extra>
    class_ &def_readonly(const char *name, const D C::*pm, const Extra& ...extra) {
        cpp_function fget([pm](const C &c) -> const D &{ return c.*pm; }, is_method(*this));
        def_property_readonly(name, fget, return_value_policy::reference_internal, extra...);
        return *this;
    }

    template <typename D, typename... Extra>
    class_ &def_readwrite_static(const char *name, D *pm, const Extra& ...extra) {
        cpp_function fget([pm](object) -> const D &{ return *pm; }, scope(*this)),
                     fset([pm](object, const D &value) { *pm = value; }, scope(*this));
        def_property_static(name, fget, fset, return_value_policy::reference, extra...);
        return *this;
    }

    template <typename D, typename... Extra>
    class_ &def_readonly_static(const char *name, const D *pm, const Extra& ...extra) {
        cpp_function fget([pm](object) -> const D &{ return *pm; }, scope(*this));
        def_property_readonly_static(name, fget, return_value_policy::reference, extra...);
        return *this;
    }

    template <typename... Extra>
    class_ &def_property_readonly(const char *name, const cpp_function &fget, const Extra& ...extra) {
        def_property(name, fget, cpp_function(), extra...);
        return *this;
    }

    template <typename... Extra>
    class_ &def_property_readonly_static(const char *name, const cpp_function &fget, const Extra& ...extra) {
        def_property_static(name, fget, cpp_function(), extra...);
        return *this;
    }

    template <typename... Extra>
    class_ &def_property(const char *name, const cpp_function &fget, const cpp_function &fset, const Extra& ...extra) {
        return def_property_static(name, fget, fset, is_method(*this), extra...);
    }

    template <typename... Extra>
    class_ &def_property_static(const char *name, const cpp_function &fget, const cpp_function &fset, const Extra& ...extra) {
        auto rec_fget = get_function_record(fget), rec_fset = get_function_record(fset);
        char *doc_prev = rec_fget->doc; /* 'extra' field may include a property-specific documentation string */
        detail::process_attributes<Extra...>::init(extra..., rec_fget);
        if (rec_fget->doc && rec_fget->doc != doc_prev) {
            free(doc_prev);
            rec_fget->doc = strdup(rec_fget->doc);
        }
        if (rec_fset) {
            doc_prev = rec_fset->doc;
            detail::process_attributes<Extra...>::init(extra..., rec_fset);
            if (rec_fset->doc && rec_fset->doc != doc_prev) {
                free(doc_prev);
                rec_fset->doc = strdup(rec_fset->doc);
            }
        }
        pybind11::str doc_obj = pybind11::str(rec_fget->doc ? rec_fget->doc : "");
        object property(
            PyObject_CallFunctionObjArgs((PyObject *) &PyProperty_Type, fget.ptr() ? fget.ptr() : Py_None,
                                         fset.ptr() ? fset.ptr() : Py_None, Py_None, doc_obj.ptr(), nullptr), false);
        if (rec_fget->class_)
            attr(name) = property;
        else
            metaclass().attr(name) = property;
        return *this;
    }

    template <typename target> class_ alias() {
        auto &instances = pybind11::detail::get_internals().registered_types_cpp;
        instances[std::type_index(typeid(target))] = instances[std::type_index(typeid(type))];
        return *this;
    }
private:
    /// Initialize holder object, variant 1: object derives from enable_shared_from_this
    template <typename T>
    static void init_holder_helper(instance_type *inst, const holder_type * /* unused */, const std::enable_shared_from_this<T> * /* dummy */) {
        try {
            new (&inst->holder) holder_type(std::static_pointer_cast<type>(inst->value->shared_from_this()));
        } catch (const std::bad_weak_ptr &) {
            new (&inst->holder) holder_type(inst->value);
        }
    }

    /// Initialize holder object, variant 2: try to construct from existing holder object, if possible
    template <typename T = holder_type,
              typename std::enable_if<std::is_copy_constructible<T>::value, int>::type = 0>
    static void init_holder_helper(instance_type *inst, const holder_type *holder_ptr, const void * /* dummy */) {
        if (holder_ptr)
            new (&inst->holder) holder_type(*holder_ptr);
        else
            new (&inst->holder) holder_type(inst->value);
    }

    /// Initialize holder object, variant 3: holder is not copy constructible (e.g. unique_ptr), always initialize from raw pointer
    template <typename T = holder_type,
              typename std::enable_if<!std::is_copy_constructible<T>::value, int>::type = 0>
    static void init_holder_helper(instance_type *inst, const holder_type * /* unused */, const void * /* dummy */) {
        new (&inst->holder) holder_type(inst->value);
    }

    /// Initialize holder object of an instance, possibly given a pointer to an existing holder
    static void init_holder(PyObject *inst_, const void *holder_ptr) {
        auto inst = (instance_type *) inst_;
        init_holder_helper(inst, (const holder_type *) holder_ptr, inst->value);
        inst->constructed = true;
    }

    static void dealloc(PyObject *inst_) {
        instance_type *inst = (instance_type *) inst_;
        if (inst->owned) {
            if (inst->constructed)
                inst->holder.~holder_type();
            else
                ::operator delete(inst->value);
        }
        generic_type::dealloc((detail::instance<void> *) inst);
    }

    static detail::function_record *get_function_record(handle h) {
        h = detail::get_function(h);
        return h ? (detail::function_record *) capsule(
               PyCFunction_GetSelf(h.ptr()), true) : nullptr;
    }
};

/// Binds C++ enumerations and enumeration classes to Python
template <typename Type> class enum_ : public class_<Type> {
public:
    template <typename... Extra>
    enum_(const handle &scope, const char *name, const Extra&... extra)
      : class_<Type>(scope, name, extra...), m_parent(scope) {
        auto entries = new std::unordered_map<int, const char *>();
        this->def("__repr__", [name, entries](Type value) -> std::string {
            auto it = entries->find((int) value);
            return std::string(name) + "." +
                ((it == entries->end()) ? std::string("???")
                                        : std::string(it->second));
        });
        this->def("__init__", [](Type& value, int i) { value = (Type)i; });
        this->def("__init__", [](Type& value, int i) { new (&value) Type((Type) i); });
        this->def("__int__", [](Type value) { return (int) value; });
        this->def("__eq__", [](const Type &value, Type *value2) { return value2 && value == *value2; });
        this->def("__ne__", [](const Type &value, Type *value2) { return !value2 || value != *value2; });
        this->def("__hash__", [](const Type &value) { return (int) value; });
        m_entries = entries;
    }

    /// Export enumeration entries into the parent scope
    void export_values() {
        PyObject *dict = ((PyTypeObject *) this->m_ptr)->tp_dict;
        PyObject *key, *value;
        ssize_t pos = 0;
        while (PyDict_Next(dict, &pos, &key, &value))
            if (PyObject_IsInstance(value, this->m_ptr))
                m_parent.attr(key) = value;
    }

    /// Add an enumeration entry
    enum_& value(char const* name, Type value) {
        this->attr(name) = pybind11::cast(value, return_value_policy::copy);
        (*m_entries)[(int) value] = name;
        return *this;
    }
private:
    std::unordered_map<int, const char *> *m_entries;
    handle m_parent;
};

NAMESPACE_BEGIN(detail)
template <typename... Args> struct init {
    template <typename Base, typename Holder, typename... Extra> void execute(pybind11::class_<Base, Holder> &class_, const Extra&... extra) const {
        /// Function which calls a specific C++ in-place constructor
        class_.def("__init__", [](Base *instance, Args... args) { new (instance) Base(args...); }, extra...);
    }
};

PYBIND11_NOINLINE inline void keep_alive_impl(int Nurse, int Patient, handle args, handle ret) {
    /* Clever approach based on weak references taken from Boost.Python */
    handle nurse  (Nurse   > 0 ? PyTuple_GetItem(args.ptr(), Nurse   - 1) : ret.ptr());
    handle patient(Patient > 0 ? PyTuple_GetItem(args.ptr(), Patient - 1) : ret.ptr());

    if (!nurse || !patient)
        pybind11_fail("Could not activate keep_alive!");

    if (patient.ptr() == Py_None)
        return; /* Nothing to keep alive */

    cpp_function disable_lifesupport(
        [patient](handle weakref) { patient.dec_ref(); weakref.dec_ref(); });

    weakref wr(nurse, disable_lifesupport);

    patient.inc_ref(); /* reference patient and leak the weak reference */
    (void) wr.release();
}

template <typename Iterator> struct iterator_state { Iterator it, end; };

NAMESPACE_END(detail)

template <typename... Args> detail::init<Args...> init() { return detail::init<Args...>(); }

template <typename Iterator,
          typename ValueType = decltype(*std::declval<Iterator>()),
          typename... Extra>
iterator make_iterator(Iterator first, Iterator last, Extra &&... extra) {
    typedef detail::iterator_state<Iterator> state;

    if (!detail::get_type_info(typeid(state))) {
        class_<state>(handle(), "")
            .def("__iter__", [](state &s) -> state& { return s; })
            .def("__next__", [](state &s) -> ValueType {
                if (s.it == s.end)
                    throw stop_iteration();
                return *s.it++;
            }, return_value_policy::reference_internal, std::forward<Extra>(extra)...);
    }

    return (iterator) cast(state { first, last });
}

template <typename Type, typename... Extra> iterator make_iterator(Type &value, Extra&&... extra) {
    return make_iterator(std::begin(value), std::end(value), extra...);
}

template <typename InputType, typename OutputType> void implicitly_convertible() {
    auto implicit_caster = [](PyObject *obj, PyTypeObject *type) -> PyObject * {
        if (!detail::type_caster<InputType>().load(obj, false))
            return nullptr;
        tuple args(1);
        args[0] = obj;
        PyObject *result = PyObject_Call((PyObject *) type, args.ptr(), nullptr);
        if (result == nullptr)
            PyErr_Clear();
        return result;
    };
    auto &registered_types = detail::get_internals().registered_types_cpp;
    auto it = registered_types.find(std::type_index(typeid(OutputType)));
    if (it == registered_types.end())
        pybind11_fail("implicitly_convertible: Unable to find type " + type_id<OutputType>());
    ((detail::type_info *) it->second)->implicit_conversions.push_back(implicit_caster);
}

#if defined(WITH_THREAD)

/* The functions below essentially reproduce the PyGILState_* API using a RAII
 * pattern, but there are a few important differences:
 *
 * 1. When acquiring the GIL from an non-main thread during the finalization
 *    phase, the GILState API blindly terminates the calling thread, which
 *    is often not what is wanted. This API does not do this.
 *
 * 2. The gil_scoped_release function can optionally cut the relationship
 *    of a PyThreadState and its associated thread, which allows moving it to
 *    another thread (this is a fairly rare/advanced use case).
 *
 * 3. The reference count of an acquired thread state can be controlled. This
 *    can be handy to prevent cases where callbacks issued from an external
 *    thread would otherwise constantly construct and destroy thread state data
 *    structures.
 *
 * See the Python bindings of NanoGUI (http://github.com/wjakob/nanogui) for an
 * example which uses features 2 and 3 to migrate the Python thread of
 * execution to another thread (to run the event loop on the original thread,
 * in this case).
 */

class gil_scoped_acquire {
public:
    PYBIND11_NOINLINE gil_scoped_acquire() {
        auto const &internals = detail::get_internals();
        tstate = (PyThreadState *) PyThread_get_key_value(internals.tstate);

        if (!tstate) {
            tstate = PyThreadState_New(internals.istate);
            #if !defined(NDEBUG)
                if (!tstate)
                    pybind11_fail("scoped_acquire: could not create thread state!");
            #endif
            tstate->gilstate_counter = 0;
            #if PY_MAJOR_VERSION < 3
                PyThread_delete_key_value(internals.tstate);
            #endif
            PyThread_set_key_value(internals.tstate, tstate);
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

    void inc_ref() {
        ++tstate->gilstate_counter;
    }

    PYBIND11_NOINLINE void dec_ref() {
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
            PyThread_delete_key_value(detail::get_internals().tstate);
            release = false;
        }
    }

    PYBIND11_NOINLINE ~gil_scoped_acquire() {
        dec_ref();
        if (release)
           PyEval_SaveThread();
    }
private:
    PyThreadState *tstate = nullptr;
    bool release = true;
};

class gil_scoped_release {
public:
    gil_scoped_release(bool disassoc = false) : disassoc(disassoc) {
        tstate = PyEval_SaveThread();
        if (disassoc) {
            auto key = detail::get_internals().tstate;
            #if PY_MAJOR_VERSION < 3
                PyThread_delete_key_value(key);
            #else
                PyThread_set_key_value(key, nullptr);
            #endif
        }
    }
    ~gil_scoped_release() {
        if (!tstate)
            return;
        PyEval_RestoreThread(tstate);
        if (disassoc) {
            auto key = detail::get_internals().tstate;
            #if PY_MAJOR_VERSION < 3
                PyThread_delete_key_value(key);
            #endif
            PyThread_set_key_value(key, tstate);
        }
    }
private:
    PyThreadState *tstate;
    bool disassoc;
};
#else
class gil_scoped_acquire { };
class gil_scoped_release { };
#endif

inline function get_overload(const void *this_ptr, const char *name)  {
    handle py_object = detail::get_object_handle(this_ptr);
    if (!py_object)
        return function();
    handle type = py_object.get_type();
    auto key = std::make_pair(type.ptr(), name);

    /* Cache functions that aren't overloaded in Python to avoid
       many costly Python dictionary lookups below */
    auto &cache = detail::get_internals().inactive_overload_cache;
    if (cache.find(key) != cache.end())
        return function();

    function overload = (function) py_object.attr(name);
    if (overload.is_cpp_function()) {
        cache.insert(key);
        return function();
    }

    /* Don't call dispatch code if invoked from overridden function */
    PyFrameObject *frame = PyThreadState_Get()->frame;
    if (frame && (std::string) pybind11::handle(frame->f_code->co_name).str() == name &&
        frame->f_code->co_argcount > 0) {
        PyFrame_FastToLocals(frame);
        PyObject *self_caller = PyDict_GetItem(
            frame->f_locals, PyTuple_GET_ITEM(frame->f_code->co_varnames, 0));
        if (self_caller == py_object.ptr())
            return function();
    }
    return overload;
}

#define PYBIND11_OVERLOAD_INT(ret_type, name, ...) { \
        pybind11::gil_scoped_acquire gil; \
        pybind11::function overload = pybind11::get_overload(this, name); \
        if (overload) \
            return overload(__VA_ARGS__).template cast<ret_type>();  }

#define PYBIND11_OVERLOAD_NAME(ret_type, cname, name, fn, ...) \
    PYBIND11_OVERLOAD_INT(ret_type, name, __VA_ARGS__) \
    return cname::fn(__VA_ARGS__)

#define PYBIND11_OVERLOAD_PURE_NAME(ret_type, cname, name, fn, ...) \
    PYBIND11_OVERLOAD_INT(ret_type, name, __VA_ARGS__) \
    pybind11::pybind11_fail("Tried to call pure virtual function \"" #cname "::" name "\"");

#define PYBIND11_OVERLOAD(ret_type, cname, fn, ...) \
    PYBIND11_OVERLOAD_NAME(ret_type, cname, #fn, fn, __VA_ARGS__)

#define PYBIND11_OVERLOAD_PURE(ret_type, cname, fn, ...) \
    PYBIND11_OVERLOAD_PURE_NAME(ret_type, cname, #fn, fn, __VA_ARGS__)

NAMESPACE_END(pybind11)

#if defined(_MSC_VER)
#  pragma warning(pop)
#elif defined(__ICC) || defined(__INTEL_COMPILER)
#  pragma warning(pop)
#elif defined(__GNUG__) && !defined(__clang__)
#  pragma GCC diagnostic pop
#endif
