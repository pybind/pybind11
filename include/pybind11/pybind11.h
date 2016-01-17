/*
    pybind11/pybind11.h: Main header file of the C++11 python binding generator library

    Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#pragma warning(disable: 4800) // warning C4800: 'int': forcing value to bool 'true' or 'false' (performance warning)
#pragma warning(disable: 4996) // warning C4996: The POSIX name for this item is deprecated. Instead, use the ISO C and C++ conformant name
#pragma warning(disable: 4100) // warning C4100: Unreferenced formal parameter
#pragma warning(disable: 4512) // warning C4512: Assignment operator was implicitly defined as deleted
#elif defined(__GNUG__) and !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-parameter"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

#include "cast.h"

NAMESPACE_BEGIN(pybind11)

template <typename T> struct arg_t;

/// Annotation for keyword arguments
struct arg {
    arg(const char *name) : name(name) { }
    template <typename T> arg_t<T> operator=(const T &value);
    const char *name;
};

/// Annotation for keyword arguments with default values
template <typename T> struct arg_t : public arg {
    arg_t(const char *name, const T &value, const char *descr = nullptr)
        : arg(name), value(value), descr(descr) { }
    T value;
    const char *descr;
};

template <typename T> arg_t<T> arg::operator=(const T &value) { return arg_t<T>(name, value); }

/// Annotation for methods
struct is_method { PyObject *class_; is_method(object *o) : class_(o->ptr()) { } };

/// Annotation for documentation
struct doc { const char *value; doc(const char *value) : value(value) { } };

/// Annotation for function names
struct name { const char *value; name(const char *value) : value(value) { } };

/// Annotation for function siblings
struct sibling { PyObject *value; sibling(handle value) : value(value.ptr()) { } };

/// Wraps an arbitrary C++ function/method/lambda function/.. into a callable Python object
class cpp_function : public function {
private:
    /// Linked list of function overloads
    struct function_entry {
        /// Function name and user-specified documentation string
        char *name = nullptr, *doc = nullptr; /* why no C++ strings? They generate heavier code.. */
        /// Human-readable version of the function signature
        char *signature = nullptr;
        /// List of registered keyword arguments
        std::vector<detail::argument_entry> args;
        /// Pointer to lambda function which converts arguments and performs the actual call
        PyObject * (*impl) (function_entry *, PyObject *, PyObject *) = nullptr;
        /// Storage for the wrapped function pointer and captured data, if any
        void *data = nullptr;
        /// Pointer to custom destructor for 'data' (if needed)
        void (*free_data) (void *ptr) = nullptr;
        /// Return value policy associated with this function
        return_value_policy policy = return_value_policy::automatic;
        /// True if name == '__init__'
        bool is_constructor = false;
        /// Python method object
        PyMethodDef *def = nullptr;
        /// Pointer to class (if this is method)
        PyObject *class_ = nullptr;
        /// Pointer to first registered function in overload chain
        PyObject *sibling = nullptr;
        /// Pointer to next overload
        function_entry *next = nullptr;
    };

    function_entry *m_entry;

    /// Picks a suitable return value converter from cast.h
    template <typename T> using return_value_caster =
        detail::type_caster<typename std::conditional<
            std::is_void<T>::value, detail::void_type, typename detail::decay<T>::type>::type>;

    /// Picks a suitable argument value converter from cast.h
    template <typename... T> using arg_value_caster =
        detail::type_caster<typename std::tuple<T...>>;

    template <typename... T> static void process_extras(const std::tuple<T...> &args, function_entry *entry) {
        process_extras(args, entry, typename detail::make_index_sequence<sizeof...(T)>::type());
    }

    template <typename... T, size_t ... Index> static void process_extras(const std::tuple<T...> &args,
            function_entry *entry, detail::index_sequence<Index...>) {
        int unused[] = { 0, (process_extra(std::get<Index>(args), entry), 0)... };
        (void) unused;
    }

    static void process_extra(const char *doc, function_entry *entry) { entry->doc = (char *) doc; }
    static void process_extra(const pybind11::doc &d, function_entry *entry) { entry->doc = (char *) d.value; }
    static void process_extra(const pybind11::name &n, function_entry *entry) { entry->name = (char *) n.value; }
    static void process_extra(const pybind11::return_value_policy p, function_entry *entry) { entry->policy = p; }
    static void process_extra(const pybind11::sibling s, function_entry *entry) { entry->sibling = s.value; }
    static void process_extra(const pybind11::is_method &m, function_entry *entry) { entry->class_ = m.class_; }
    static void process_extra(const pybind11::arg &a, function_entry *entry) {
        if (entry->class_ && entry->args.empty())
            entry->args.emplace_back("self", nullptr, nullptr);
        entry->args.emplace_back(a.name, nullptr, nullptr);
    }

    template <typename T>
    static void process_extra(const pybind11::arg_t<T> &a, function_entry *entry) {
        if (entry->class_ && entry->args.empty())
            entry->args.emplace_back("self", nullptr, nullptr);

        PyObject *obj = detail::type_caster<typename detail::decay<T>::type>::cast(
                a.value, return_value_policy::automatic, nullptr);

        if (obj == nullptr)
            throw std::runtime_error("arg(): could not convert default keyword "
                                     "argument into a Python object (type not "
                                     "registered yet?)");

        entry->args.emplace_back(a.name, a.descr, obj);
    }
public:
    cpp_function() { }

    /// Vanilla function pointers
    template <typename Return, typename... Args, typename... Extra>
    cpp_function(Return (*f)(Args...), Extra&&... extra) {
        using detail::descr;
        m_entry = new function_entry();
        m_entry->data = (void *) f;

        typedef arg_value_caster<Args...> cast_in;
        typedef return_value_caster<Return> cast_out;

        m_entry->impl = [](function_entry *entry, PyObject *pyArgs, PyObject *parent) -> PyObject * {
            cast_in args;
            if (!args.load(pyArgs, true))
                return (PyObject *) 1; /* Special return code: try next overload */
            return cast_out::cast(args.template call<Return>((Return (*)(Args...)) entry->data), entry->policy, parent);
        };

        process_extras(std::make_tuple(std::forward<Extra>(extra)...), m_entry);
        PYBIND11_DESCR signature = cast_in::name() + detail::_(" -> ") + cast_out::name();
        initialize(signature.text(), signature.types(), sizeof...(Args));
    }

    /// Delegating helper constructor to deal with lambda functions
    template <typename Func, typename... Extra> cpp_function(Func &&f, Extra&&... extra) {
        initialize(std::forward<Func>(f),
                   (typename detail::remove_class<decltype(
                       &std::remove_reference<Func>::type::operator())>::type *) nullptr,
                   std::forward<Extra>(extra)...);
    }

    /// Class methods (non-const)
    template <typename Return, typename Class, typename... Arg, typename... Extra> cpp_function(
            Return (Class::*f)(Arg...), Extra&&... extra) {
        initialize([f](Class *c, Arg... args) -> Return { return (c->*f)(args...); },
                   (Return (*) (Class *, Arg...)) nullptr, std::forward<Extra>(extra)...);
    }

    /// Class methods (const)
    template <typename Return, typename Class, typename... Arg, typename... Extra> cpp_function(
            Return (Class::*f)(Arg...) const, Extra&&... extra) {
        initialize([f](const Class *c, Arg... args) -> Return { return (c->*f)(args...); },
                   (Return (*)(const Class *, Arg ...)) nullptr, std::forward<Extra>(extra)...);
    }

    /// Return the function name
    const char *name() const { return m_entry->name; }

private:
    /// Functors, lambda functions, etc.
    template <typename Func, typename Return, typename... Args, typename... Extra>
    void initialize(Func &&f, Return (*)(Args...), Extra&&... extra) {
        using detail::descr;

        struct capture { typename std::remove_reference<Func>::type f; };

        m_entry = new function_entry();
        m_entry->data = new capture { std::forward<Func>(f) };

        if (!std::is_trivially_destructible<Func>::value)
            m_entry->free_data = [](void *ptr) { delete (capture *) ptr; };
        else
            m_entry->free_data = operator delete;

        typedef arg_value_caster<Args...> cast_in;
        typedef return_value_caster<Return> cast_out;

        m_entry->impl = [](function_entry *entry, PyObject *pyArgs, PyObject *parent) -> PyObject *{
            cast_in args;
            if (!args.load(pyArgs, true))
                return (PyObject *) 1; /* Special return code: try next overload */
            return cast_out::cast(args.template call<Return>(((capture *) entry->data)->f), entry->policy, parent);
        };

        process_extras(std::make_tuple(std::forward<Extra>(extra)...), m_entry);
        PYBIND11_DESCR signature = cast_in::name() + detail::_(" -> ") + cast_out::name();
        initialize(signature.text(), signature.types(), sizeof...(Args));
    }

    static PyObject *dispatcher(PyObject *self, PyObject *args, PyObject *kwargs) {
        function_entry *overloads = (function_entry *) PyCapsule_GetPointer(self, nullptr),
                       *it = overloads;
        int nargs = (int) PyTuple_Size(args),
            nkwargs = kwargs ? (int) PyDict_Size(kwargs) : 0;
        PyObject *parent = nargs > 0 ? PyTuple_GetItem(args, 0) : nullptr,
                 *result = (PyObject *) 1;
        try {
            for (; it != nullptr; it = it->next) {
                object args_(args, true);
                int kwargs_consumed = 0;

                if (nargs < (int) it->args.size()) {
                    args_ = object(PyTuple_New(it->args.size()), false);
                    for (int i = 0; i < nargs; ++i) {
                        PyObject *item = PyTuple_GET_ITEM(args, i);
                        Py_INCREF(item);
                        PyTuple_SET_ITEM(args_.ptr(), i, item);
                    }
                    int arg_ctr = 0;
                    for (auto const &it2 : it->args) {
                        int index = arg_ctr++;
                        if (PyTuple_GET_ITEM(args_.ptr(), index))
                            continue;
                        PyObject *value = nullptr;
                        if (kwargs)
                            value = PyDict_GetItemString(kwargs, it2.name);
                        if (value)
                            kwargs_consumed++;
                        else if (it2.value)
                            value = it2.value;
                        if (value) {
                            Py_INCREF(value);
                            PyTuple_SET_ITEM(args_.ptr(), index, value);
                        } else {
                            kwargs_consumed = -1; /* definite failure */
                            break;
                        }
                    }
                }

                if (kwargs_consumed == nkwargs)
                    result = it->impl(it, args_.ptr(), parent);

                if (result != (PyObject *) 1)
                    break;
            }
        } catch (const error_already_set &)      {                                                 return nullptr;
        } catch (const index_error &e)           { PyErr_SetString(PyExc_IndexError,    e.what()); return nullptr;
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
        if (result == (PyObject *) 1) {
            std::string msg = "Incompatible function arguments. The "
                              "following argument types are supported:\n";
            int ctr = 0;
            for (function_entry *it2 = overloads; it2 != nullptr; it2 = it2->next) {
                msg += "    "+ std::to_string(++ctr) + ". ";
                msg += it2->signature;
                msg += "\n";
            }
            PyErr_SetString(PyExc_TypeError, msg.c_str());
            return nullptr;
        } else if (result == nullptr) {
            std::string msg = "Unable to convert function return value to a "
                              "Python type! The signature was\n\t";
            msg += it->signature;
            PyErr_SetString(PyExc_TypeError, msg.c_str());
            return nullptr;
        } else {
            if (overloads->is_constructor) {
                PyObject *inst = PyTuple_GetItem(args, 0);
                const detail::type_info *type_info =
                    capsule(PyObject_GetAttrString((PyObject *) Py_TYPE(inst),
                                const_cast<char *>("__pybind11__")), false);
                type_info->init_holder(inst);
            }
            return result;
        }
    }

    static void destruct(function_entry *entry) {
        while (entry) {
            function_entry *next = entry->next;
            delete entry->def;
            if (entry->free_data)
                entry->free_data(entry->data);
            std::free((char *) entry->name);
            std::free((char *) entry->doc);
            std::free((char *) entry->signature);
            for (auto &arg: entry->args) {
                std::free((char *) arg.name);
                std::free((char *) arg.descr);
                Py_XDECREF(arg.value);
            }
            delete entry;
            entry = next;
        }
    }

    void initialize(const char *text, const std::type_info * const * types, int args) {
        /* Create copies of all referenced C-style strings */
        m_entry->name = strdup(m_entry->name ? m_entry->name : "");
        if (m_entry->doc) m_entry->doc = strdup(m_entry->doc);
        for (auto &a: m_entry->args) {
            if (a.name)
                a.name = strdup(a.name);
            if (a.descr)
                a.descr = strdup(a.descr);
            else if (a.value)
                a.descr = strdup(((std::string) ((object) handle(a.value).attr("__repr__")).call().str()).c_str());
        }
        auto const &registered_types = detail::get_internals().registered_types;

        /* Generate a proper function signature */
        std::string signature;
        size_t type_depth = 0, char_index = 0, type_index = 0, arg_index = 0;
        while (true) {
            char c = text[char_index++];
            if (c == '\0')
                break;

            if (c == '{') {
                if (type_depth == 1 && arg_index < m_entry->args.size()) {
                    signature += m_entry->args[arg_index].name;
                    signature += " : ";
                }
                ++type_depth;
            } else if (c == '}') {
                --type_depth;
                if (type_depth == 1 && arg_index < m_entry->args.size()) {
                    if (m_entry->args[arg_index].descr) {
                        signature += " = ";
                        signature += m_entry->args[arg_index].descr;
                    }
                    arg_index++;
                }
            } else if (c == '%') {
                const std::type_info *t = types[type_index++];
                if (!t) 
                    throw std::runtime_error("Internal error while parsing type signature (1)");
                auto it = registered_types.find(t);
                if (it != registered_types.end()) {
                    signature += it->second.type->tp_name;
                } else {
                    std::string tname(t->name());
                    detail::clean_type_id(tname);
                    signature += tname;
                }
            } else {
                signature += c;
            }
        }
        if (type_depth != 0 && types[type_index ] != nullptr)
            throw std::runtime_error("Internal error while parsing type signature (2)");

        #if !defined(PYBIND11_CPP14)
            delete[] types;
            delete[] text;
        #endif

#if PY_MAJOR_VERSION < 3
        if (strcmp(m_entry->name, "__next__") == 0) {
            free(m_entry->name);
            m_entry->name = strdup("next");
        }
#endif

        if (!m_entry->args.empty() && (int) m_entry->args.size() != args)
            throw std::runtime_error(
                "cpp_function(): function \"" + std::string(m_entry->name) + "\" takes " +
                std::to_string(args) + " arguments, but " + std::to_string(m_entry->args.size()) +
                " pybind11::arg entries were specified!");

        m_entry->is_constructor = !strcmp(m_entry->name, "__init__");
        m_entry->signature = strdup(signature.c_str());
        m_entry->args.shrink_to_fit();

#if PY_MAJOR_VERSION < 3
        if (m_entry->sibling && PyMethod_Check(m_entry->sibling))
            m_entry->sibling = PyMethod_GET_FUNCTION(m_entry->sibling);
#endif

        function_entry *s_entry = nullptr, *entry = m_entry;
        if (m_entry->sibling && PyCFunction_Check(m_entry->sibling)) {
            capsule entry_capsule(PyCFunction_GetSelf(m_entry->sibling), true);
            s_entry = (function_entry *) entry_capsule;
            if (s_entry->class_ != m_entry->class_)
                s_entry = nullptr; /* Overridden method, don't append to parent class overloads */
        }

        if (!s_entry) { /* No existing overload was found, create a function object */
            m_entry->def = new PyMethodDef();
            memset(m_entry->def, 0, sizeof(PyMethodDef));
            m_entry->def->ml_name = m_entry->name;
            m_entry->def->ml_meth = reinterpret_cast<PyCFunction>(*dispatcher);
            m_entry->def->ml_flags = METH_VARARGS | METH_KEYWORDS;
            capsule entry_capsule(m_entry, [](PyObject *o) { destruct((function_entry *) PyCapsule_GetPointer(o, nullptr)); });
            m_ptr = PyCFunction_New(m_entry->def, entry_capsule.ptr());
            if (!m_ptr)
                throw std::runtime_error("cpp_function::cpp_function(): Could not allocate function object");
        } else {
            m_ptr = m_entry->sibling;
            inc_ref();
            entry = s_entry;
            while (s_entry->next)
                s_entry = s_entry->next;
            s_entry->next = m_entry;
        }

        std::string signatures;
        int index = 0;
        function_entry *it = entry;
        while (it) { /* Create pydoc entry including all function signatures and docstrings of the overload chain */
            if (s_entry)
                signatures += std::to_string(++index) + ". ";
            signatures += "Signature : ";
            signatures += it->signature;
            signatures += "\n";
            if (it->doc && strlen(it->doc) > 0) {
                signatures += "\n";
                signatures += it->doc;
                signatures += "\n";
            }
            if (it->next)
                signatures += "\n";
            it = it->next;
        }
        PyCFunctionObject *func = (PyCFunctionObject *) m_ptr;
        if (func->m_ml->ml_doc)
            free((char *) func->m_ml->ml_doc);
        func->m_ml->ml_doc = strdup(signatures.c_str());
        if (entry->class_) {
#if PY_MAJOR_VERSION >= 3
            m_ptr = PyInstanceMethod_New(m_ptr);
#else
            m_ptr = PyMethod_New(m_ptr, nullptr, entry->class_);
#endif
            if (!m_ptr)
                throw std::runtime_error("cpp_function::cpp_function(): Could not allocate instance method object");
            Py_DECREF(func);
        }
    }
};

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
            throw std::runtime_error("Internal error in module::module()");
        inc_ref();
    }

    template <typename Func, typename... Extra>
    module &def(const char *name_, Func &&f, Extra&& ... extra) {
        cpp_function func(std::forward<Func>(f), name(name_),
                          sibling((handle) attr(name_)), std::forward<Extra>(extra)...);
        func.inc_ref(); /* The following line steals a reference to 'func' */
        PyModule_AddObject(ptr(), name_, func.ptr());
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
            throw std::runtime_error("Module \"" + std::string(name) + "\" not found!");
        return module(obj, false);
    }
};

NAMESPACE_BEGIN(detail)
/// Basic support for creating new Python heap types
class custom_type : public object {
public:
    PYBIND11_OBJECT_DEFAULT(custom_type, object, PyType_Check)

    custom_type(object &scope, const char *name_, const std::type_info *tinfo,
                size_t type_size, size_t instance_size,
                void (*init_holder)(PyObject *), const destructor &dealloc,
                PyObject *parent, const char *doc) {
        PyHeapTypeObject *type = (PyHeapTypeObject*) PyType_Type.tp_alloc(&PyType_Type, 0);
#if PY_MAJOR_VERSION >= 3
        PyObject *name = PyUnicode_FromString(name_);
#else
        PyObject *name = PyString_FromString(name_);
#endif
        if (type == nullptr || name == nullptr)
            throw std::runtime_error("Internal error in custom_type::custom_type()");
        Py_INCREF(name);
        std::string full_name(name_);

        pybind11::str scope_name = (object) scope.attr("__name__"),
                    module_name = (object) scope.attr("__module__");

        if (scope_name.check())
            full_name =  std::string(scope_name) + "." + full_name;
        if (module_name.check())
            full_name =  std::string(module_name) + "." + full_name;

        type->ht_name = name;
#if PY_MAJOR_VERSION >= 3
        type->ht_qualname = name;
#endif
        type->ht_type.tp_name = strdup(full_name.c_str());
        type->ht_type.tp_basicsize = instance_size;
        type->ht_type.tp_init = (initproc) init;
        type->ht_type.tp_new = (newfunc) new_instance;
        type->ht_type.tp_dealloc = dealloc;
        type->ht_type.tp_flags |=
            Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;
        type->ht_type.tp_flags &= ~Py_TPFLAGS_HAVE_GC;
#if PY_MAJOR_VERSION < 3
        type->ht_type.tp_flags |= Py_TPFLAGS_CHECKTYPES;
#endif
        type->ht_type.tp_as_number = &type->as_number;
        type->ht_type.tp_as_sequence = &type->as_sequence;
        type->ht_type.tp_as_mapping = &type->as_mapping;
        type->ht_type.tp_base = (PyTypeObject *) parent;
        if (doc) {
            size_t size = strlen(doc)+1;
            type->ht_type.tp_doc = (char *) PyObject_MALLOC(size);
            memcpy((void *) type->ht_type.tp_doc, doc, size);
        }
        Py_XINCREF(parent);

        if (PyType_Ready(&type->ht_type) < 0)
            throw std::runtime_error("Internal error in custom_type::custom_type()");
        m_ptr = (PyObject *) type;

        /* Needed by pydoc */
        attr("__module__") = scope_name;

        auto &registered_types = get_internals().registered_types;
        auto &type_info = registered_types[tinfo];
        type_info.type = (PyTypeObject *) m_ptr;
        type_info.type_size = type_size;
        type_info.init_holder = init_holder;
        attr("__pybind11__") = capsule(&type_info);

        scope.attr(name) = *this;
    }

protected:
    /* Allocate a metaclass on demand (for static properties) */
    handle metaclass() {
        auto &ht_type = ((PyHeapTypeObject *) m_ptr)->ht_type;
#if PY_MAJOR_VERSION >= 3
        auto &ob_type = ht_type.ob_base.ob_base.ob_type;
#else
        auto &ob_type = ht_type.ob_type;
#endif

        if (ob_type == &PyType_Type) {
            std::string name_ = std::string(ht_type.tp_name) + "_meta";
            PyHeapTypeObject *type = (PyHeapTypeObject*) PyType_Type.tp_alloc(&PyType_Type, 0);
            PyObject *name = PyUnicode_FromString(name_.c_str());
            if (type == nullptr || name == nullptr)
                throw std::runtime_error("Internal error in custom_type::metaclass()");
            Py_INCREF(name);
            type->ht_name = name;
#if PY_MAJOR_VERSION >= 3
            type->ht_qualname = name;
#endif
            type->ht_type.tp_name = strdup(name_.c_str());
            type->ht_type.tp_base = &PyType_Type;
            type->ht_type.tp_flags |= Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE;
            type->ht_type.tp_flags &= ~Py_TPFLAGS_HAVE_GC;
            if (PyType_Ready(&type->ht_type) < 0)
                throw std::runtime_error("Internal error in custom_type::metaclass()");
            ob_type = (PyTypeObject *) type;
            Py_INCREF(type);
        }
        return handle((PyObject *) ob_type);
    }

    static int init(void *self, PyObject *, PyObject *) {
        std::string msg = std::string(Py_TYPE(self)->tp_name) + ": No constructor defined!";
        PyErr_SetString(PyExc_TypeError, msg.c_str());
        return -1;
    }

    static PyObject *new_instance(PyTypeObject *type, PyObject *, PyObject *) {
        const detail::type_info *type_info = capsule(
            PyObject_GetAttrString((PyObject *) type, const_cast<char*>("__pybind11__")), false);
        instance<void> *self = (instance<void> *) PyType_GenericAlloc(type, 0);
        self->value = ::operator new(type_info->type_size);
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
                    throw std::runtime_error("Deallocating unregistered instance!");
                registered_instances.erase(it);
            }
            Py_XDECREF(self->parent);
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
        auto info = ((detail::type_info *) capsule(attr("__pybind11__")));
        info->get_buffer = get_buffer;
        info->get_buffer_data = get_buffer_data;
    }

    static int getbuffer(PyObject *obj, Py_buffer *view, int flags) {
        auto const &typeinfo = ((detail::type_info *) capsule(handle(obj).attr("__pybind11__")));

        if (view == nullptr || obj == nullptr || !typeinfo || !typeinfo->get_buffer) {
            PyErr_SetString(PyExc_BufferError, "Internal error");
            return -1;
        }
        memset(view, 0, sizeof(Py_buffer));
        buffer_info *info = typeinfo->get_buffer(obj, typeinfo->get_buffer_data);
        view->obj = obj;
        view->ndim = 1;
        view->internal = info;
        view->buf = info->ptr;
        view->itemsize = info->itemsize;
        view->len = view->itemsize;
        for (auto s : info->shape)
            view->len *= s;
        if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT)
            view->format = const_cast<char *>(info->format.c_str());
        if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
            view->ndim = info->ndim;
            view->strides = (ssize_t *) &info->strides[0];
            view->shape = (ssize_t *) &info->shape[0];
        }
        Py_INCREF(view->obj);
        return 0;
    }

    static void releasebuffer(PyObject *, Py_buffer *view) { delete (buffer_info *) view->internal; }
};

/* Forward declarations */
enum op_id : int;
enum op_type : int;
struct undefined_t;
template <op_id id, op_type ot, typename L = undefined_t, typename R = undefined_t> struct op_;
template <typename... Args> struct init;
NAMESPACE_END(detail)

template <typename type, typename holder_type = std::unique_ptr<type>> class class_ : public detail::custom_type {
public:
    typedef detail::instance<type, holder_type> instance_type;

    PYBIND11_OBJECT(class_, detail::custom_type, PyType_Check)

    class_(object &scope, const char *name, const char *doc = nullptr)
        : detail::custom_type(scope, name, &typeid(type), sizeof(type),
                              sizeof(instance_type), init_holder, dealloc,
                              nullptr, doc) { }

    class_(object &scope, const char *name, object &parent,
           const char *doc = nullptr)
        : detail::custom_type(scope, name, &typeid(type), sizeof(type),
                              sizeof(instance_type), init_holder, dealloc,
                              parent.ptr(), doc) { }

    template <typename Func, typename... Extra>
    class_ &def(const char *name_, Func&& f, Extra&&... extra) {
        cpp_function cf(std::forward<Func>(f), name(name_),
                        sibling(attr(name_)), is_method(this),
                        std::forward<Extra>(extra)...);
        attr(cf.name()) = cf;
        return *this;
    }

    template <typename Func, typename... Extra> class_ &
    def_static(const char *name_, Func f, Extra&&... extra) {
        cpp_function cf(std::forward<Func>(f), name(name_),
                        sibling(attr(name_)),
                        std::forward<Extra>(extra)...);
        attr(cf.name()) = cf;
        return *this;
    }

    template <detail::op_id id, detail::op_type ot, typename L, typename R, typename... Extra>
    class_ &def(const detail::op_<id, ot, L, R> &op, Extra&&... extra) {
        op.template execute<type>(*this, std::forward<Extra>(extra)...);
        return *this;
    }

    template <detail::op_id id, detail::op_type ot, typename L, typename R, typename... Extra>
    class_ & def_cast(const detail::op_<id, ot, L, R> &op, Extra&&... extra) {
        op.template execute_cast<type>(*this, std::forward<Extra>(extra)...);
        return *this;
    }

    template <typename... Args, typename... Extra>
    class_ &def(const detail::init<Args...> &init, Extra&&... extra) {
        init.template execute<type>(*this, std::forward<Extra>(extra)...);
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
    class_ &def_readwrite(const char *name, D C::*pm, Extra&&... extra) {
        cpp_function fget([pm](const C &c) -> const D &{ return c.*pm; },
                          return_value_policy::reference_internal,
                          is_method(this), extra...),
                     fset([pm](C &c, const D &value) { c.*pm = value; },
                          is_method(this), extra...);
        def_property(name, fget, fset);
        return *this;
    }

    template <typename C, typename D, typename... Extra>
    class_ &def_readonly(const char *name, const D C::*pm, Extra&& ...extra) {
        cpp_function fget([pm](const C &c) -> const D &{ return c.*pm; },
                          return_value_policy::reference_internal,
                          is_method(this), std::forward<Extra>(extra)...);
        def_property_readonly(name, fget);
        return *this;
    }

    template <typename D, typename... Extra>
    class_ &def_readwrite_static(const char *name, D *pm, Extra&& ...extra) {
        cpp_function fget([pm](object) -> const D &{ return *pm; }, nullptr,
                          return_value_policy::reference_internal, extra...),
                     fset([pm](object, const D &value) { *pm = value; }, extra...);
        def_property_static(name, fget, fset);
        return *this;
    }

    template <typename D, typename... Extra>
    class_ &def_readonly_static(const char *name, const D *pm, Extra&& ...extra) {
        cpp_function fget([pm](object) -> const D &{ return *pm; }, nullptr,
                          return_value_policy::reference_internal, std::forward<Extra>(extra)...);
        def_property_readonly_static(name, fget);
        return *this;
    }

    class_ &def_property_readonly(const char *name, const cpp_function &fget, const char *doc = nullptr) {
        def_property(name, fget, cpp_function(), doc);
        return *this;
    }

    class_ &def_property_readonly_static(const char *name, const cpp_function &fget, const char *doc = nullptr) {
        def_property_static(name, fget, cpp_function(), doc);
        return *this;
    }

    class_ &def_property(const char *name, const cpp_function &fget, const cpp_function &fset, const char *doc = nullptr) {
        object doc_obj = doc ? pybind11::str(doc) : (object) const_cast<cpp_function&>(fget).attr("__doc__");
        object property(
            PyObject_CallFunction((PyObject *)&PyProperty_Type,
                                  const_cast<char *>("OOOO"), fget.ptr() ? fget.ptr() : Py_None,
                                  fset.ptr() ? fset.ptr() : Py_None, Py_None, doc_obj.ptr()), false);
        attr(name) = property;
        return *this;
    }

    class_ &def_property_static(const char *name, const cpp_function &fget, const cpp_function &fset, const char *doc = nullptr) {
        object doc_obj = doc ? pybind11::str(doc) : (object) const_cast<cpp_function&>(fget).attr("__doc__");
        object property(
            PyObject_CallFunction((PyObject *)&PyProperty_Type,
                                  const_cast<char *>("OOOs"), fget.ptr() ? fget.ptr() : Py_None,
                                  fset.ptr() ? fset.ptr() : Py_None, Py_None, doc_obj.ptr()), false);
        metaclass().attr(name) = property;
        return *this;
    }

    template <typename target> class_ alias() {
        auto &instances = pybind11::detail::get_internals().registered_types;
        instances[&typeid(target)] = instances[&typeid(type)];
        return *this;
    }
private:
    template <typename T = holder_type,
              typename std::enable_if<!std::is_same<std::shared_ptr<type>, T>::value, int>::type = 0>
    static void init_holder(PyObject *inst_) {
        instance_type *inst = (instance_type *) inst_;
        new (&inst->holder) holder_type(inst->value);
        inst->constructed = true;
    }

    template <typename T = holder_type,
              typename std::enable_if<std::is_same<std::shared_ptr<type>, T>::value, int>::type = 0>
    static void init_holder(PyObject *inst_) {
        instance_type *inst = (instance_type *) inst_;
        try {
            new (&inst->holder) holder_type(
                inst->value->shared_from_this());
        } catch (const std::bad_weak_ptr &) {
            new (&inst->holder) holder_type(inst->value);
        }
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
        custom_type::dealloc((detail::instance<void> *) inst);
    }
};

/// Binds C++ enumerations and enumeration classes to Python
template <typename Type> class enum_ : public class_<Type> {
public:
    enum_(object &scope, const char *name, const char *doc = nullptr)
      : class_<Type>(scope, name, doc), m_parent(scope) {
        auto entries = new std::unordered_map<int, const char *>();
        this->def("__repr__", [name, entries](Type value) -> std::string {
            auto it = entries->find((int) value);
            return std::string(name) + "." +
                ((it == entries->end()) ? std::string("???")
                                        : std::string(it->second));
        });
        this->def("__int__", [](Type value) { return (int) value; });
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
    object &m_parent;
};

NAMESPACE_BEGIN(detail)
template <typename... Args> struct init {
    template <typename Base, typename Holder, typename... Extra> void execute(pybind11::class_<Base, Holder> &class_, Extra&&... extra) const {
        /// Function which calls a specific C++ in-place constructor
        class_.def("__init__", [](Base *instance, Args... args) { new (instance) Base(args...); }, std::forward<Extra>(extra)...);
    }
};
NAMESPACE_END(detail)

template <typename... Args> detail::init<Args...> init() { return detail::init<Args...>(); };

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
    auto & registered_types = detail::get_internals().registered_types;
    auto it = registered_types.find(&typeid(OutputType));
    if (it == registered_types.end())
        throw std::runtime_error("implicitly_convertible: Unable to find type " + type_id<OutputType>());
    it->second.implicit_conversions.push_back(implicit_caster);
}

inline void init_threading() { PyEval_InitThreads(); }

class gil_scoped_acquire {
    PyGILState_STATE state;
public:
    inline gil_scoped_acquire() { state = PyGILState_Ensure(); }
    inline ~gil_scoped_acquire() { PyGILState_Release(state); }
};

class gil_scoped_release {
    PyThreadState *state;
public:
    inline gil_scoped_release() { state = PyEval_SaveThread(); }
    inline ~gil_scoped_release() { PyEval_RestoreThread(state); }
};

inline function get_overload(const void *this_ptr, const char *name)  {
    handle py_object = detail::get_object_handle(this_ptr);
    if (!py_object)
        return function();
    handle type = py_object.get_type();
    auto key = std::make_pair(type.ptr(), name);

    /* Cache functions that aren't overloaded in python to avoid
       many costly dictionary lookups in Python */
    auto &cache = detail::get_internals().inactive_overload_cache;
    if (cache.find(key) != cache.end())
        return function();

    function overload = (function) py_object.attr(name);
    if (overload.is_cpp_function()) {
        cache.insert(key);
        return function();
    }

    PyFrameObject *frame = PyThreadState_Get()->frame;
    pybind11::str caller = pybind11::handle(frame->f_code->co_name).str();
    if ((std::string) caller == name)
        return function();
    return overload;
}

#define PYBIND11_OVERLOAD_INT(ret_type, class_name, name, ...) { \
        pybind11::gil_scoped_acquire gil; \
        pybind11::function overload = pybind11::get_overload(this, #name); \
        if (overload) \
            return overload.call(__VA_ARGS__).cast<ret_type>();  }

#define PYBIND11_OVERLOAD(ret_type, class_name, name, ...) \
    PYBIND11_OVERLOAD_INT(ret_type, class_name, name, __VA_ARGS__) \
    return class_name::name(__VA_ARGS__)

#define PYBIND11_OVERLOAD_PURE(ret_type, class_name, name, ...) \
    PYBIND11_OVERLOAD_INT(ret_type, class_name, name, __VA_ARGS__) \
    throw std::runtime_error("Tried to call pure virtual function \"" #name "\"");

NAMESPACE_END(pybind11)

#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUG__) and !defined(__clang__)
#pragma GCC diagnostic pop
#endif

