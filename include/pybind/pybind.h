/*
    pybind/pybind.h: Main header file of the C++11 python binding generator library

    Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#if !defined(__PYBIND_H)
#define __PYBIND_H

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#pragma warning(disable: 4800) // warning C4800: 'int': forcing value to bool 'true' or 'false' (performance warning)
#pragma warning(disable: 4996) // warning C4996: The POSIX name for this item is deprecated. Instead, use the ISO C and C++ conformant name
#pragma warning(disable: 4100) // warning C4100: Unreferenced formal parameter
#pragma warning(disable: 4512) // warning C4512: Assignment operator was implicitly defined as deleted
#endif

#include "cast.h"

NAMESPACE_BEGIN(pybind)

class function : public object {
private:
    struct function_entry {
        std::function<PyObject* (PyObject *)> impl;
        std::string signature, doc;
        bool is_constructor;
        function_entry *next = nullptr;
    };
public:
    PYTHON_OBJECT_DEFAULT(function, object, PyFunction_Check)

    template <typename Func>
    function(const char *name, Func _func, bool is_method,
             function overload_sibling = function(), const char *doc = nullptr,
             return_value_policy policy = return_value_policy::automatic) {
        /* Function traits extracted from the template type 'Func' */
        typedef mpl::function_traits<Func> f_traits;

        /* Suitable input and output casters */
        typedef typename detail::type_caster<typename f_traits::args_type> cast_in;
        typedef typename detail::type_caster<typename mpl::normalize_type<typename f_traits::return_type>::type> cast_out;
        typename f_traits::f_type func = f_traits::cast(_func);

        auto impl = [func, policy](PyObject *pyArgs) -> PyObject *{
            cast_in args;
            if (!args.load(pyArgs, true))
                return nullptr;
            PyObject *parent = policy != return_value_policy::reference_internal
                ? nullptr : PyTuple_GetItem(pyArgs, 0);
            return cast_out::cast(
                f_traits::dispatch(func, (typename f_traits::args_type) args),
                policy, parent);
        };

        /* Linked list of function call handlers (for overloading) */
        function_entry *entry = new function_entry();
        entry->impl = impl;
        entry->signature = std::string(name) + cast_in::name() + std::string(" -> ") + cast_out::name();
        entry->is_constructor = !strcmp(name, "__init__");
        if (doc) entry->doc = doc;

        install_function(name, entry, is_method, overload_sibling);
    }

private:
    static PyObject *dispatcher(PyObject *self, PyObject *args, PyObject * /* kwargs */) {
        function_entry *overloads = (function_entry *) PyCapsule_GetPointer(self, nullptr);
        PyObject *result = nullptr;
        try {
            for (function_entry *it = overloads; it != nullptr; it = it->next) {
                if ((result = it->impl(args)) != nullptr)
                    break;
            }
        } catch (const error_already_set &) {                                               return nullptr;
        } catch (const index_error &e)    { PyErr_SetString(PyExc_IndexError,    e.what()); return nullptr;
        } catch (const stop_iteration &e) { PyErr_SetString(PyExc_StopIteration, e.what()); return nullptr;
        } catch (const std::exception &e) { PyErr_SetString(PyExc_RuntimeError,  e.what()); return nullptr;
        } catch (...) {
            PyErr_SetString(PyExc_RuntimeError, "Caught an unknown exception!");
            return nullptr;
        }
        if (result) {
            if (overloads->is_constructor) {
                PyObject *inst = PyTuple_GetItem(args, 0);
                const detail::type_info *type_info =
                    capsule(PyObject_GetAttrString((PyObject *) Py_TYPE(inst),
                                const_cast<char *>("__pybind__")), false);
                type_info->init_holder(inst);
            }
            return result;
        } else {
            std::string signatures = "Incompatible function arguments. The "
                                     "following argument types are supported:\n";
            int ctr = 0;
            for (function_entry *it = overloads; it != nullptr; it = it->next) {
                signatures += "    "+ std::to_string(++ctr) + ". ";
                signatures += it->signature;
                signatures += "\n";
            }
            PyErr_SetString(PyExc_TypeError, signatures.c_str());
            return nullptr;
        }
    }

    void install_function(const char *name, function_entry *entry, bool is_method, function overload_sibling) {
        if (!overload_sibling.ptr() || !PyCFunction_Check(overload_sibling.ptr())) {
            PyMethodDef *def = new PyMethodDef();
            memset(def, 0, sizeof(PyMethodDef));
            def->ml_name = strdup(name);
            def->ml_meth = reinterpret_cast<PyCFunction>(*dispatcher);
            def->ml_flags = METH_VARARGS | METH_KEYWORDS;
            capsule entry_capsule(entry);
            m_ptr = PyCFunction_New(def, entry_capsule.ptr());
            if (!m_ptr)
                throw std::runtime_error("function::function(): Could not allocate function object");
        } else {
            m_ptr = overload_sibling.ptr();
            inc_ref();
            capsule entry_capsule(PyCFunction_GetSelf(m_ptr), true);
            function_entry *parent = (function_entry *) entry_capsule, *backup = parent;
            while (parent->next)
                parent = parent->next;
            parent->next = entry;
            entry = backup;
        }
        std::string signatures;
        while (entry) { /* Create pydoc entry */
            signatures += "Signature : " + std::string(entry->signature) + "\n";
            if (!entry->doc.empty())
                signatures += "\n" + std::string(entry->doc) + "\n";
            if (entry->next)
                signatures += "\n";
            entry = entry->next;
        }
        PyCFunctionObject *func = (PyCFunctionObject *) m_ptr;
        if (func->m_ml->ml_doc)
            std::free((char *) func->m_ml->ml_doc);
        func->m_ml->ml_doc = strdup(signatures.c_str());
        if (is_method) {
            m_ptr = PyInstanceMethod_New(m_ptr);
            if (!m_ptr)
                throw std::runtime_error("function::function(): Could not allocate instance method object");
            Py_DECREF(func);
        }
    }
};

class module : public object {
public:
    PYTHON_OBJECT_DEFAULT(module, object, PyModule_Check)

    module(const char *name, const char *doc = nullptr) {
        PyModuleDef *def = new PyModuleDef();
        memset(def, 0, sizeof(PyModuleDef));
        def->m_name = name;
        def->m_doc = doc;
        def->m_size = -1;
        Py_INCREF(def);
        m_ptr = PyModule_Create(def);
        if (m_ptr == nullptr)
            throw std::runtime_error("Internal error in module::module()");
        inc_ref();
    }

    template <typename Func> module& def(const char *name, Func f, const char *doc = nullptr) {
        function func(name, f, false, (function) attr(name), doc);
        func.inc_ref(); /* The following line steals a reference to 'func' */
        PyModule_AddObject(ptr(), name, func.ptr());
        return *this;
    }

    module def_submodule(const char *name) {
        std::string full_name = std::string(PyModule_GetName(m_ptr))
            + std::string(".") + std::string(name);
        module result(PyImport_AddModule(full_name.c_str()), true);
        attr(name) = result;
        return result;
    }
};

NAMESPACE_BEGIN(detail)
/* Forward declarations */
enum op_id : int;
enum op_type : int;
struct undefined_t;
template <op_id id, op_type ot, typename L = undefined_t, typename R = undefined_t> struct op_;
template <typename ... Args> struct init;

/// Basic support for creating new Python heap types
class custom_type : public object {
public:
    PYTHON_OBJECT_DEFAULT(custom_type, object, PyType_Check)

    custom_type(object &scope, const char *name_, const std::string &type_name,
                size_t type_size, size_t instance_size,
                void (*init_holder)(PyObject *), const destructor &dealloc,
                PyObject *parent, const char *doc) {
        PyHeapTypeObject *type = (PyHeapTypeObject*) PyType_Type.tp_alloc(&PyType_Type, 0);
        PyObject *name = PyUnicode_FromString(name_);
        if (type == nullptr || name == nullptr)
            throw std::runtime_error("Internal error in custom_type::custom_type()");
        Py_INCREF(name);
        std::string full_name(name_);

        pybind::str scope_name = (object) scope.attr("__name__"),
                    module_name = (object) scope.attr("__module__");

        if (scope_name.check())
            full_name =  std::string(scope_name) + "." + full_name;
        if (module_name.check())
            full_name =  std::string(module_name) + "." + full_name;

        type->ht_name = type->ht_qualname = name;
        type->ht_type.tp_name = strdup(full_name.c_str());
        type->ht_type.tp_basicsize = instance_size;
        type->ht_type.tp_doc = doc;
        type->ht_type.tp_init = (initproc) init;
        type->ht_type.tp_new = (newfunc) new_instance;
        type->ht_type.tp_dealloc = dealloc;
        type->ht_type.tp_flags |=
            Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;
        type->ht_type.tp_flags &= ~Py_TPFLAGS_HAVE_GC;
        type->ht_type.tp_as_number = &type->as_number;
        type->ht_type.tp_as_sequence = &type->as_sequence;
        type->ht_type.tp_as_mapping = &type->as_mapping;
        type->ht_type.tp_base = (PyTypeObject *) parent;
        Py_XINCREF(parent);

        if (PyType_Ready(&type->ht_type) < 0)
            throw std::runtime_error("Internal error in custom_type::custom_type()");
        m_ptr = (PyObject *) type;

        /* Needed by pydoc */
        if (((module &) scope).check())
            attr("__module__") = scope_name;

        auto &type_info = detail::get_internals().registered_types[type_name];
        type_info.type = (PyTypeObject *) m_ptr;
        type_info.type_size = type_size;
        type_info.init_holder = init_holder;
        attr("__pybind__") = capsule(&type_info);

        scope.attr(name) = *this;
    }

protected:
    /* Allocate a metaclass on demand (for static properties) */
    handle metaclass() {
        auto &ht_type = ((PyHeapTypeObject *) m_ptr)->ht_type;
        auto &ob_type = ht_type.ob_base.ob_base.ob_type;
        if (ob_type == &PyType_Type) {
            std::string name_ = std::string(ht_type.tp_name) + "_meta";
            PyHeapTypeObject *type = (PyHeapTypeObject*) PyType_Type.tp_alloc(&PyType_Type, 0);
            PyObject *name = PyUnicode_FromString(name_.c_str());
            if (type == nullptr || name == nullptr)
                throw std::runtime_error("Internal error in custom_type::metaclass()");
            Py_INCREF(name);
            type->ht_name = type->ht_qualname = name;
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
            PyObject_GetAttrString((PyObject *) type, const_cast<char*>("__pybind__")), false);
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

    void install_buffer_funcs(const std::function<buffer_info *(PyObject *)> &func) {
        PyHeapTypeObject *type = (PyHeapTypeObject*) m_ptr;
        type->ht_type.tp_as_buffer = &type->as_buffer;
        type->as_buffer.bf_getbuffer = getbuffer;
        type->as_buffer.bf_releasebuffer = releasebuffer;
        ((detail::type_info *) capsule(attr("__pybind__")))->get_buffer = func;
    }

    static int getbuffer(PyObject *obj, Py_buffer *view, int flags) {
        auto const &info_func = ((detail::type_info *) capsule(handle(obj).attr("__pybind__")))->get_buffer;
        if (view == nullptr || obj == nullptr || !info_func) {
            PyErr_SetString(PyExc_BufferError, "Internal error");
            return -1;
        }
        memset(view, 0, sizeof(Py_buffer));
        buffer_info *info = info_func(obj);
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
            view->strides = (Py_ssize_t *)&info->strides[0];
            view->shape = (Py_ssize_t *) &info->shape[0];
        }
        Py_INCREF(view->obj);
        return 0;
    }

    static void releasebuffer(PyObject *, Py_buffer *view) { delete (buffer_info *) view->internal; }
};

NAMESPACE_END(detail)

template <typename type, typename holder_type = std::unique_ptr<type>> class class_ : public detail::custom_type {
public:
    typedef detail::instance<type, holder_type> instance_type;

    PYTHON_OBJECT(class_, detail::custom_type, PyType_Check)

    class_(object &scope, const char *name, const char *doc = nullptr)
        : detail::custom_type(scope, name, type_id<type>(), sizeof(type),
                              sizeof(instance_type), init_holder, dealloc,
                              nullptr, doc) { }

    class_(object &scope, const char *name, object &parent,
           const char *doc = nullptr)
        : detail::custom_type(scope, name, type_id<type>(), sizeof(type),
                              sizeof(instance_type), init_holder, dealloc,
                              parent.ptr(), doc) { }

    template <typename Func>
    class_ &def(const char *name, Func f, const char *doc = nullptr,
                return_value_policy policy = return_value_policy::automatic) {
        attr(name) = function(name, f, true, (function) attr(name), doc, policy);
        return *this;
    }

    template <typename Func> class_ &
    def_static(const char *name, Func f, const char *doc = nullptr,
               return_value_policy policy = return_value_policy::automatic) {
        attr(name) = function(name, f, false, (function) attr(name), doc, policy);
        return *this;
    }

    template <detail::op_id id, detail::op_type ot, typename L, typename R>
    class_ &def(const detail::op_<id, ot, L, R> &op, const char *doc = nullptr,
                return_value_policy policy = return_value_policy::automatic) {
        op.template execute<type>(*this, doc, policy);
        return *this;
    }

    template <detail::op_id id, detail::op_type ot, typename L, typename R> class_ &
    def_cast(const detail::op_<id, ot, L, R> &op, const char *doc = nullptr,
             return_value_policy policy = return_value_policy::automatic) {
        op.template execute_cast<type>(*this, doc, policy);
        return *this;
    }

    template <typename... Args>
    class_ &def(const detail::init<Args...> &init, const char *doc = nullptr) {
        init.template execute<type>(*this, doc);
        return *this;
    }

    class_& def_buffer(const std::function<buffer_info(type&)> &func) {
        install_buffer_funcs([func](PyObject *obj) -> buffer_info* {
            detail::type_caster<type> caster;
            if (!caster.load(obj, false))
                return nullptr;
            return new buffer_info(func(caster));
        });
        return *this;
    }

    template <typename C, typename D>
    class_ &def_readwrite(const char *name, D C::*pm,
                          const char *doc = nullptr) {
        function fget("", [=](C * ptr) -> D & { return ptr->*pm; }, true,
                      function(), doc, return_value_policy::reference_internal),
                 fset("", [=](C *ptr, const D &value) { ptr->*pm = value; }, true, function(), doc);
        def_property(name, fget, fset, doc);
        return *this;
    }

    template <typename C, typename D>
    class_ &def_readonly(const char *name, const D C::*pm,
                         const char *doc = nullptr) {
        function fget("", [=](C * ptr) -> const D & { return ptr->*pm; }, true,
                      function(), doc, return_value_policy::reference_internal);
        def_property(name, fget, doc);
        return *this;
    }

    template <typename D>
    class_ &def_readwrite_static(const char *name, D *pm,
                                 const char *doc = nullptr) {
        function fget("", [=](object) -> D & { return *pm; }, true),
                 fset("", [=](object, const D &value) { *pm = value; }, true);
        def_property_static(name, fget, fset, doc);
        return *this;
    }

    template <typename D>
    class_ &def_readonly_static(const char *name, const D *pm,
                                const char *doc = nullptr) {
        function fget("", [=](object) -> const D & { return *pm; }, true);
        def_property_static(name, fget, doc);
        return *this;
    }

    class_ &def_property(const char *name, const function &fget,
                         const char *doc = nullptr) {
        def_property(name, fget, function(), doc);
        return *this;
    }

    class_ &def_property_static(const char *name, const function &fget,
                                const char *doc = nullptr) {
        def_property_static(name, fget, function(), doc);
        return *this;
    }

    class_ &def_property(const char *name, const function &fget,
                         const function &fset, const char *doc = nullptr) {
        object property(
            PyObject_CallFunction((PyObject *)&PyProperty_Type,
                                  const_cast<char *>("OOOs"), fget.ptr() ? fget.ptr() : Py_None,
                                  fset.ptr() ? fset.ptr() : Py_None, Py_None, doc), false);
        attr(name) = property;
        return *this;
    }

    class_ &def_property_static(const char *name, const function &fget,
                                const function &fset,
                                const char *doc = nullptr) {
        object property(
            PyObject_CallFunction((PyObject *)&PyProperty_Type,
                                  const_cast<char *>("OOOs"), fget.ptr() ? fget.ptr() : Py_None,
                                  fset.ptr() ? fset.ptr() : Py_None, Py_None, doc), false);
        metaclass().attr(name) = property;
        return *this;
    }
private:
    static void init_holder(PyObject *inst_) {
        instance_type *inst = (instance_type *) inst_;
        new (&inst->holder) holder_type(inst->value);
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
        this->def("__str__", [name, entries](Type value) -> std::string {
            auto it = entries->find(value);
            return std::string(name) + "." +
                ((it == entries->end()) ? std::string("???")
                                        : std::string(it->second));
        });
        m_entries = entries;
    }

    /// Export enumeration entries into the parent scope
    void export_values() {
        PyObject *dict = ((PyTypeObject *) this->m_ptr)->tp_dict;
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(dict, &pos, &key, &value))
            if (PyObject_IsInstance(value, this->m_ptr))
                m_parent.attr(key) = value;
    }

    /// Add an enumeration entry
    enum_& value(char const* name, Type value) {
        this->attr(name) = pybind::cast(value, return_value_policy::copy);
        (*m_entries)[(int) value] = name;
        return *this;
    }
private:
    std::unordered_map<int, const char *> *m_entries;
    object &m_parent;
};

NAMESPACE_BEGIN(detail)
template <typename ... Args> struct init {
    template <typename Base, typename Holder> void execute(pybind::class_<Base, Holder> &class_, const char *doc) const {
        /// Function which calls a specific C++ in-place constructor
        class_.def("__init__", [](Base *instance, Args... args) { new (instance) Base(args...); }, doc);
    }
};
NAMESPACE_END(detail)

template <typename... Args> detail::init<Args...> init() { return detail::init<Args...>(); };

template <typename InputType, typename OutputType> void implicitly_convertible() {
    auto implicit_caster = [](PyObject *obj, PyTypeObject *type) -> PyObject *{
        if (!detail::type_caster<InputType>().load(obj, false))
            return nullptr;
        tuple args(1);
        args[0] = obj;
        PyObject *result = PyObject_Call((PyObject *) type, args.ptr(), nullptr);
        if (result == nullptr)
            PyErr_Clear();
        return result;
    };
    std::string output_type_name = type_id<OutputType>();
    auto & registered_types = detail::get_internals().registered_types;
    auto it = registered_types.find(output_type_name);
    if (it == registered_types.end())
        throw std::runtime_error("implicitly_convertible: Unable to find type " + output_type_name);
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

NAMESPACE_END(pybind)

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#undef PYTHON_OBJECT
#undef PYTHON_OBJECT_DEFAULT

#endif /* __PYBIND_H */
