/*
    pybind11/pybind11.h: Main header file of the C++11 python
    binding generator library

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#if defined(__INTEL_COMPILER)
#  pragma warning push
#  pragma warning disable 68    // integer conversion resulted in a change of sign
#  pragma warning disable 186   // pointless comparison of unsigned integer with zero
#  pragma warning disable 878   // incompatible exception specifications
#  pragma warning disable 1334  // the "template" keyword used for syntactic disambiguation may only be used within a template
#  pragma warning disable 1682  // implicit conversion of a 64-bit integral type to a smaller integral type (potential portability problem)
#  pragma warning disable 1786  // function "strdup" was declared deprecated
#  pragma warning disable 1875  // offsetof applied to non-POD (Plain Old Data) types is nonstandard
#  pragma warning disable 2196  // warning #2196: routine is both "inline" and "noinline"
#elif defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4100) // warning C4100: Unreferenced formal parameter
#  pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#  pragma warning(disable: 4512) // warning C4512: Assignment operator was implicitly defined as deleted
#  pragma warning(disable: 4800) // warning C4800: 'int': forcing value to bool 'true' or 'false' (performance warning)
#  pragma warning(disable: 4996) // warning C4996: The POSIX name for this item is deprecated. Instead, use the ISO C and C++ conformant name
#  pragma warning(disable: 4702) // warning C4702: unreachable code
#  pragma warning(disable: 4522) // warning C4522: multiple assignment operators specified
#  pragma warning(disable: 4505) // warning C4505: 'PySlice_GetIndicesEx': unreferenced local function has been removed (PyPy only)
#elif defined(__GNUG__) && !defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wunused-but-set-parameter"
#  pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#  pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#  pragma GCC diagnostic ignored "-Wstrict-aliasing"
#  pragma GCC diagnostic ignored "-Wattributes"
#  if __GNUC__ >= 7
#    pragma GCC diagnostic ignored "-Wnoexcept-type"
#  endif
#endif

#include "attr.h"
#include "gil.h"
#include "options.h"
#include "detail/class.h"
#include "detail/function.h"
#include "detail/init.h"

#include <memory>
#include <vector>
#include <string>
#include <utility>

#if defined(__GNUG__) && !defined(__clang__)
#  include <cxxabi.h>
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
/// Wrapper for Python extension modules
class module_ : public object {
public:
    PYBIND11_OBJECT_DEFAULT(module_, object, PyModule_Check)

    /// Create a new top-level Python module with the given name and docstring
    PYBIND11_DEPRECATED("Use PYBIND11_MODULE or module_::create_extension_module instead")
    explicit module_(const char *name, const char *doc = nullptr) {
#if PY_MAJOR_VERSION >= 3
        *this = create_extension_module(name, doc, new PyModuleDef());
#else
        *this = create_extension_module(name, doc, nullptr);
#endif
    }

    /** \rst
        Create Python binding for a new function within the module scope. ``Func``
        can be a plain C++ function, a function pointer, or a lambda function. For
        details on the ``Extra&& ... extra`` argument, see section :ref:`extras`.
    \endrst */
    template <typename Func, typename... Extra>
    module_ &def(const char *name_, Func &&f, const Extra& ... extra) {
        cpp_function func(std::forward<Func>(f), name(name_), scope(*this),
                          sibling(getattr(*this, name_, none())), extra...);
        // NB: allow overwriting here because cpp_function sets up a chain with the intention of
        // overwriting (and has already checked internally that it isn't overwriting non-functions).
        add_object(name_, func, true /* overwrite */);
        return *this;
    }

    /** \rst
        Create and return a new Python submodule with the given name and docstring.
        This also works recursively, i.e.

        .. code-block:: cpp

            py::module_ m("example", "pybind11 example plugin");
            py::module_ m2 = m.def_submodule("sub", "A submodule of 'example'");
            py::module_ m3 = m2.def_submodule("subsub", "A submodule of 'example.sub'");
    \endrst */
    module_ def_submodule(const char *name, const char *doc = nullptr) {
        std::string full_name = std::string(PyModule_GetName(m_ptr))
            + std::string(".") + std::string(name);
        auto result = reinterpret_borrow<module_>(PyImport_AddModule(full_name.c_str()));
        if (doc && options::show_user_defined_docstrings())
            result.attr("__doc__") = pybind11::str(doc);
        attr(name) = result;
        return result;
    }

    /// Import and return a module or throws `error_already_set`.
    static module_ import(const char *name) {
        PyObject *obj = PyImport_ImportModule(name);
        if (!obj)
            throw error_already_set();
        return reinterpret_steal<module_>(obj);
    }

    /// Reload the module or throws `error_already_set`.
    void reload() {
        PyObject *obj = PyImport_ReloadModule(ptr());
        if (!obj)
            throw error_already_set();
        *this = reinterpret_steal<module_>(obj);
    }

    /** \rst
        Adds an object to the module using the given name.  Throws if an object with the given name
        already exists.

        ``overwrite`` should almost always be false: attempting to overwrite objects that pybind11 has
        established will, in most cases, break things.
    \endrst */
    PYBIND11_NOINLINE void add_object(const char *name, handle obj, bool overwrite = false) {
        if (!overwrite && hasattr(*this, name))
            pybind11_fail("Error during initialization: multiple incompatible definitions with name \"" +
                    std::string(name) + "\"");

        PyModule_AddObject(ptr(), name, obj.inc_ref().ptr() /* steals a reference */);
    }

#if PY_MAJOR_VERSION >= 3
    using module_def = PyModuleDef;
#else
    struct module_def {};
#endif

    /** \rst
        Create a new top-level module that can be used as the main module of a C extension.

        For Python 3, ``def`` should point to a statically allocated module_def.
        For Python 2, ``def`` can be a nullptr and is completely ignored.
    \endrst */
    static module_ create_extension_module(const char *name, const char *doc, module_def *def) {
#if PY_MAJOR_VERSION >= 3
        // module_def is PyModuleDef
        def = new (def) PyModuleDef {  // Placement new (not an allocation).
            /* m_base */     PyModuleDef_HEAD_INIT,
            /* m_name */     name,
            /* m_doc */      options::show_user_defined_docstrings() ? doc : nullptr,
            /* m_size */     -1,
            /* m_methods */  nullptr,
            /* m_slots */    nullptr,
            /* m_traverse */ nullptr,
            /* m_clear */    nullptr,
            /* m_free */     nullptr
        };
        auto m = PyModule_Create(def);
#else
        // Ignore module_def *def; only necessary for Python 3
        (void) def;
        auto m = Py_InitModule3(name, nullptr, options::show_user_defined_docstrings() ? doc : nullptr);
#endif
        if (m == nullptr) {
            if (PyErr_Occurred())
                throw error_already_set();
            pybind11_fail("Internal error in module_::create_extension_module()");
        }
        // TODO: Should be reinterpret_steal for Python 3, but Python also steals it again when returned from PyInit_...
        //       For Python 2, reinterpret_borrow is correct.
        return reinterpret_borrow<module_>(m);
    }
};

// When inside a namespace (or anywhere as long as it's not the first item on a line),
// C++20 allows "module" to be used. This is provided for backward compatibility, and for
// simplicity, if someone wants to use py::module for example, that is perfectly safe.
using module = module_;

/// \ingroup python_builtins
/// Return a dictionary representing the global variables in the current execution frame,
/// or ``__main__.__dict__`` if there is no frame (usually when the interpreter is embedded).
inline dict globals() {
    PyObject *p = PyEval_GetGlobals();
    return reinterpret_borrow<dict>(p ? p : module_::import("__main__").attr("__dict__").ptr());
}

PYBIND11_NAMESPACE_BEGIN(detail)
/// Generic support for creating new Python heap types
class generic_type : public object {
public:
    PYBIND11_OBJECT_DEFAULT(generic_type, object, PyType_Check)
protected:
    void initialize(const type_record &rec) {
        if (rec.scope && hasattr(rec.scope, "__dict__") && rec.scope.attr("__dict__").contains(rec.name))
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

    /// Helper function which tags all parents of a type using mult. inheritance
    void mark_parents_nonsimple(PyTypeObject *value) {
        auto t = reinterpret_borrow<tuple>(value->tp_bases);
        for (handle h : t) {
            auto tinfo2 = get_type_info((PyTypeObject *) h.ptr());
            if (tinfo2)
                tinfo2->simple_type = false;
            mark_parents_nonsimple((PyTypeObject *) h.ptr());
        }
    }

    void install_buffer_funcs(
            buffer_info *(*get_buffer)(PyObject *, void *),
            void *get_buffer_data) {
        auto *type = (PyHeapTypeObject*) m_ptr;
        auto tinfo = detail::get_type_info(&type->ht_type);

        if (!type->ht_type.tp_as_buffer)
            pybind11_fail(
                "To be able to register buffer protocol support for the type '" +
                get_fully_qualified_tp_name(tinfo->type) +
                "' the associated class<>(..) invocation must "
                "include the pybind11::buffer_protocol() annotation!");

        tinfo->get_buffer = get_buffer;
        tinfo->get_buffer_data = get_buffer_data;
    }

    // rec_func must be set for either fget or fset.
    void def_property_static_impl(const char *name,
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
};

/// Set the pointer to operator new if it exists. The cast is needed because it can be overloaded.
template <typename T, typename = void_t<decltype(static_cast<void *(*)(size_t)>(T::operator new))>>
void set_operator_new(type_record *r) { r->operator_new = &T::operator new; }

template <typename> void set_operator_new(...) { }

template <typename T, typename SFINAE = void> struct has_operator_delete : std::false_type { };
template <typename T> struct has_operator_delete<T, void_t<decltype(static_cast<void (*)(void *)>(T::operator delete))>>
    : std::true_type { };
template <typename T, typename SFINAE = void> struct has_operator_delete_size : std::false_type { };
template <typename T> struct has_operator_delete_size<T, void_t<decltype(static_cast<void (*)(void *, size_t)>(T::operator delete))>>
    : std::true_type { };
/// Call class-specific delete if it exists or global otherwise. Can also be an overload set.
template <typename T, enable_if_t<has_operator_delete<T>::value, int> = 0>
void call_operator_delete(T *p, size_t, size_t) { T::operator delete(p); }
template <typename T, enable_if_t<!has_operator_delete<T>::value && has_operator_delete_size<T>::value, int> = 0>
void call_operator_delete(T *p, size_t s, size_t) { T::operator delete(p, s); }

inline void call_operator_delete(void *p, size_t s, size_t a) {
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

inline void add_class_method(object& cls, const char *name_, const cpp_function &cf) {
    cls.attr(cf.name()) = cf;
    if (strcmp(name_, "__eq__") == 0 && !cls.attr("__dict__").contains("__hash__")) {
      cls.attr("__hash__") = none();
    }
}

PYBIND11_NAMESPACE_END(detail)

/// Given a pointer to a member function, cast it to its `Derived` version.
/// Forward everything else unchanged.
template <typename /*Derived*/, typename F>
auto method_adaptor(F &&f) -> decltype(std::forward<F>(f)) { return std::forward<F>(f); }

template <typename Derived, typename Return, typename Class, typename... Args>
auto method_adaptor(Return (Class::*pmf)(Args...)) -> Return (Derived::*)(Args...) {
    static_assert(detail::is_accessible_base_of<Class, Derived>::value,
        "Cannot bind an inaccessible base class method; use a lambda definition instead");
    return pmf;
}

template <typename Derived, typename Return, typename Class, typename... Args>
auto method_adaptor(Return (Class::*pmf)(Args...) const) -> Return (Derived::*)(Args...) const {
    static_assert(detail::is_accessible_base_of<Class, Derived>::value,
        "Cannot bind an inaccessible base class method; use a lambda definition instead");
    return pmf;
}

template <typename type_, typename... options>
class class_ : public detail::generic_type {
    template <typename T> using is_holder = detail::is_holder_type<type_, T>;
    template <typename T> using is_subtype = detail::is_strict_base_of<type_, T>;
    template <typename T> using is_base = detail::is_strict_base_of<T, type_>;
    // struct instead of using here to help MSVC:
    template <typename T> struct is_valid_class_option :
        detail::any_of<is_holder<T>, is_subtype<T>, is_base<T>> {};

public:
    using type = type_;
    using type_alias = detail::exactly_one_t<is_subtype, void, options...>;
    constexpr static bool has_alias = !std::is_void<type_alias>::value;
    using holder_type = detail::exactly_one_t<is_holder, std::unique_ptr<type>, options...>;

    static_assert(detail::all_of<is_valid_class_option<options>...>::value,
            "Unknown/invalid class_ template parameters provided");

    static_assert(!has_alias || std::is_polymorphic<type>::value,
            "Cannot use an alias class with a non-polymorphic type");

    PYBIND11_OBJECT(class_, generic_type, PyType_Check)

    template <typename... Extra>
    class_(handle scope, const char *name, const Extra &... extra) {
        using namespace detail;

        // MI can only be specified via class_ template options, not constructor parameters
        static_assert(
            none_of<is_pyobject<Extra>...>::value || // no base class arguments, or:
            (   constexpr_sum(is_pyobject<Extra>::value...) == 1 && // Exactly one base
                constexpr_sum(is_base<options>::value...)   == 0 && // no template option bases
                none_of<std::is_same<multiple_inheritance, Extra>...>::value), // no multiple_inheritance attr
            "Error: multiple inheritance bases must be specified via class_ template options");

        type_record record;
        record.scope = scope;
        record.name = name;
        record.type = &typeid(type);
        record.type_size = sizeof(conditional_t<has_alias, type_alias, type>);
        record.type_align = alignof(conditional_t<has_alias, type_alias, type>&);
        record.holder_size = sizeof(holder_type);
        record.init_instance = init_instance;
        record.dealloc = dealloc;
        record.default_holder = detail::is_instantiation<std::unique_ptr, holder_type>::value;

        set_operator_new<type>(&record);

        /* Register base classes specified via template arguments to class_, if any */
        PYBIND11_EXPAND_SIDE_EFFECTS(add_base<options>(record));

        /* Process optional arguments, if any */
        process_attributes<Extra...>::init(extra..., &record);

        generic_type::initialize(record);

        if (has_alias) {
            auto &instances = record.module_local ? registered_local_types_cpp() : get_internals().registered_types_cpp;
            instances[std::type_index(typeid(type_alias))] = instances[std::type_index(typeid(type))];
        }
    }

    template <typename Base, detail::enable_if_t<is_base<Base>::value, int> = 0>
    static void add_base(detail::type_record &rec) {
        rec.add_base(typeid(Base), [](void *src) -> void * {
            return static_cast<Base *>(reinterpret_cast<type *>(src));
        });
    }

    template <typename Base, detail::enable_if_t<!is_base<Base>::value, int> = 0>
    static void add_base(detail::type_record &) { }

    template <typename Func, typename... Extra>
    class_ &def(const char *name_, Func&& f, const Extra&... extra) {
        cpp_function cf(method_adaptor<type>(std::forward<Func>(f)), name(name_), is_method(*this),
                        sibling(getattr(*this, name_, none())), extra...);
        add_class_method(*this, name_, cf);
        return *this;
    }

    template <typename Func, typename... Extra> class_ &
    def_static(const char *name_, Func &&f, const Extra&... extra) {
        static_assert(!std::is_member_function_pointer<Func>::value,
                "def_static(...) called with a non-static member function pointer");
        cpp_function cf(std::forward<Func>(f), name(name_), scope(*this),
                        sibling(getattr(*this, name_, none())), extra...);
        attr(cf.name()) = staticmethod(cf);
        return *this;
    }

    template <detail::op_id id, detail::op_type ot, typename L, typename R, typename... Extra>
    class_ &def(const detail::op_<id, ot, L, R> &op, const Extra&... extra) {
        op.execute(*this, extra...);
        return *this;
    }

    template <detail::op_id id, detail::op_type ot, typename L, typename R, typename... Extra>
    class_ & def_cast(const detail::op_<id, ot, L, R> &op, const Extra&... extra) {
        op.execute_cast(*this, extra...);
        return *this;
    }

    template <typename... Args, typename... Extra>
    class_ &def(const detail::initimpl::constructor<Args...> &init, const Extra&... extra) {
        init.execute(*this, extra...);
        return *this;
    }

    template <typename... Args, typename... Extra>
    class_ &def(const detail::initimpl::alias_constructor<Args...> &init, const Extra&... extra) {
        init.execute(*this, extra...);
        return *this;
    }

    template <typename... Args, typename... Extra>
    class_ &def(detail::initimpl::factory<Args...> &&init, const Extra&... extra) {
        std::move(init).execute(*this, extra...);
        return *this;
    }

    template <typename... Args, typename... Extra>
    class_ &def(detail::initimpl::pickle_factory<Args...> &&pf, const Extra &...extra) {
        std::move(pf).execute(*this, extra...);
        return *this;
    }

    template <typename Func>
    class_& def_buffer(Func &&func) {
        struct capture { Func func; };
        auto *ptr = new capture { std::forward<Func>(func) };
        install_buffer_funcs([](PyObject *obj, void *ptr) -> buffer_info* {
            detail::make_caster<type> caster;
            if (!caster.load(obj, false))
                return nullptr;
            return new buffer_info(((capture *) ptr)->func(caster));
        }, ptr);
        weakref(m_ptr, cpp_function([ptr](handle wr) {
            delete ptr;
            wr.dec_ref();
        })).release();
        return *this;
    }

    template <typename Return, typename Class, typename... Args>
    class_ &def_buffer(Return (Class::*func)(Args...)) {
        return def_buffer([func] (type &obj) { return (obj.*func)(); });
    }

    template <typename Return, typename Class, typename... Args>
    class_ &def_buffer(Return (Class::*func)(Args...) const) {
        return def_buffer([func] (const type &obj) { return (obj.*func)(); });
    }

    template <typename C, typename D, typename... Extra>
    class_ &def_readwrite(const char *name, D C::*pm, const Extra&... extra) {
        static_assert(std::is_same<C, type>::value || std::is_base_of<C, type>::value, "def_readwrite() requires a class member (or base class member)");
        cpp_function fget([pm](const type &c) -> const D &{ return c.*pm; }, is_method(*this)),
                     fset([pm](type &c, const D &value) { c.*pm = value; }, is_method(*this));
        def_property(name, fget, fset, return_value_policy::reference_internal, extra...);
        return *this;
    }

    template <typename C, typename D, typename... Extra>
    class_ &def_readonly(const char *name, const D C::*pm, const Extra& ...extra) {
        static_assert(std::is_same<C, type>::value || std::is_base_of<C, type>::value, "def_readonly() requires a class member (or base class member)");
        cpp_function fget([pm](const type &c) -> const D &{ return c.*pm; }, is_method(*this));
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

    /// Uses return_value_policy::reference_internal by default
    template <typename Getter, typename... Extra>
    class_ &def_property_readonly(const char *name, const Getter &fget, const Extra& ...extra) {
        return def_property_readonly(name, cpp_function(method_adaptor<type>(fget)),
                                     return_value_policy::reference_internal, extra...);
    }

    /// Uses cpp_function's return_value_policy by default
    template <typename... Extra>
    class_ &def_property_readonly(const char *name, const cpp_function &fget, const Extra& ...extra) {
        return def_property(name, fget, nullptr, extra...);
    }

    /// Uses return_value_policy::reference by default
    template <typename Getter, typename... Extra>
    class_ &def_property_readonly_static(const char *name, const Getter &fget, const Extra& ...extra) {
        return def_property_readonly_static(name, cpp_function(fget), return_value_policy::reference, extra...);
    }

    /// Uses cpp_function's return_value_policy by default
    template <typename... Extra>
    class_ &def_property_readonly_static(const char *name, const cpp_function &fget, const Extra& ...extra) {
        return def_property_static(name, fget, nullptr, extra...);
    }

    /// Uses return_value_policy::reference_internal by default
    template <typename Getter, typename Setter, typename... Extra>
    class_ &def_property(const char *name, const Getter &fget, const Setter &fset, const Extra& ...extra) {
        return def_property(name, fget, cpp_function(method_adaptor<type>(fset)), extra...);
    }
    template <typename Getter, typename... Extra>
    class_ &def_property(const char *name, const Getter &fget, const cpp_function &fset, const Extra& ...extra) {
        return def_property(name, cpp_function(method_adaptor<type>(fget)), fset,
                            return_value_policy::reference_internal, extra...);
    }

    /// Uses cpp_function's return_value_policy by default
    template <typename... Extra>
    class_ &def_property(const char *name, const cpp_function &fget, const cpp_function &fset, const Extra& ...extra) {
        return def_property_static(name, fget, fset, is_method(*this), extra...);
    }

    /// Uses return_value_policy::reference by default
    template <typename Getter, typename... Extra>
    class_ &def_property_static(const char *name, const Getter &fget, const cpp_function &fset, const Extra& ...extra) {
        return def_property_static(name, cpp_function(fget), fset, return_value_policy::reference, extra...);
    }

    /// Uses cpp_function's return_value_policy by default
    template <typename... Extra>
    class_ &def_property_static(const char *name, const cpp_function &fget, const cpp_function &fset, const Extra& ...extra) {
        static_assert( 0 == detail::constexpr_sum(std::is_base_of<arg, Extra>::value...),
                      "Argument annotations are not allowed for properties");
        auto rec_fget = get_function_record(fget), rec_fset = get_function_record(fset);
        auto *rec_active = rec_fget;
        if (rec_fget) {
           char *doc_prev = rec_fget->doc; /* 'extra' field may include a property-specific documentation string */
           detail::process_attributes<Extra...>::init(extra..., rec_fget);
           if (rec_fget->doc && rec_fget->doc != doc_prev) {
              free(doc_prev);
              rec_fget->doc = strdup(rec_fget->doc);
           }
        }
        if (rec_fset) {
            char *doc_prev = rec_fset->doc;
            detail::process_attributes<Extra...>::init(extra..., rec_fset);
            if (rec_fset->doc && rec_fset->doc != doc_prev) {
                free(doc_prev);
                rec_fset->doc = strdup(rec_fset->doc);
            }
            if (! rec_active) rec_active = rec_fset;
        }
        def_property_static_impl(name, fget, fset, rec_active);
        return *this;
    }

private:
    /// Initialize holder object, variant 1: object derives from enable_shared_from_this
    template <typename T>
    static void init_holder(detail::instance *inst, detail::value_and_holder &v_h,
            const holder_type * /* unused */, const std::enable_shared_from_this<T> * /* dummy */) {

        auto sh = std::dynamic_pointer_cast<typename holder_type::element_type>(
                detail::try_get_shared_from_this(v_h.value_ptr<type>()));
        if (sh) {
            new (std::addressof(v_h.holder<holder_type>())) holder_type(std::move(sh));
            v_h.set_holder_constructed();
        }

        if (!v_h.holder_constructed() && inst->owned) {
            new (std::addressof(v_h.holder<holder_type>())) holder_type(v_h.value_ptr<type>());
            v_h.set_holder_constructed();
        }
    }

    static void init_holder_from_existing(const detail::value_and_holder &v_h,
            const holder_type *holder_ptr, std::true_type /*is_copy_constructible*/) {
        new (std::addressof(v_h.holder<holder_type>())) holder_type(*reinterpret_cast<const holder_type *>(holder_ptr));
    }

    static void init_holder_from_existing(const detail::value_and_holder &v_h,
            const holder_type *holder_ptr, std::false_type /*is_copy_constructible*/) {
        new (std::addressof(v_h.holder<holder_type>())) holder_type(std::move(*const_cast<holder_type *>(holder_ptr)));
    }

    /// Initialize holder object, variant 2: try to construct from existing holder object, if possible
    static void init_holder(detail::instance *inst, detail::value_and_holder &v_h,
            const holder_type *holder_ptr, const void * /* dummy -- not enable_shared_from_this<T>) */) {
        if (holder_ptr) {
            init_holder_from_existing(v_h, holder_ptr, std::is_copy_constructible<holder_type>());
            v_h.set_holder_constructed();
        } else if (inst->owned || detail::always_construct_holder<holder_type>::value) {
            new (std::addressof(v_h.holder<holder_type>())) holder_type(v_h.value_ptr<type>());
            v_h.set_holder_constructed();
        }
    }

    /// Performs instance initialization including constructing a holder and registering the known
    /// instance.  Should be called as soon as the `type` value_ptr is set for an instance.  Takes an
    /// optional pointer to an existing holder to use; if not specified and the instance is
    /// `.owned`, a new holder will be constructed to manage the value pointer.
    static void init_instance(detail::instance *inst, const void *holder_ptr) {
        auto v_h = inst->get_value_and_holder(detail::get_type_info(typeid(type)));
        if (!v_h.instance_registered()) {
            register_instance(inst, v_h.value_ptr(), v_h.type);
            v_h.set_instance_registered();
        }
        init_holder(inst, v_h, (const holder_type *) holder_ptr, v_h.value_ptr<type>());
    }

    /// Deallocates an instance; via holder, if constructed; otherwise via operator delete.
    static void dealloc(detail::value_and_holder &v_h) {
        // We could be deallocating because we are cleaning up after a Python exception.
        // If so, the Python error indicator will be set. We need to clear that before
        // running the destructor, in case the destructor code calls more Python.
        // If we don't, the Python API will exit with an exception, and pybind11 will
        // throw error_already_set from the C++ destructor which is forbidden and triggers
        // std::terminate().
        error_scope scope;
        if (v_h.holder_constructed()) {
            v_h.holder<holder_type>().~holder_type();
            v_h.set_holder_constructed(false);
        }
        else {
            detail::call_operator_delete(v_h.value_ptr<type>(),
                v_h.type->type_size,
                v_h.type->type_align
            );
        }
        v_h.value_ptr() = nullptr;
    }

    static detail::function_record *get_function_record(handle h) {
        h = detail::get_function(h);
        return h ? (detail::function_record *) reinterpret_borrow<capsule>(PyCFunction_GET_SELF(h.ptr()))
                 : nullptr;
    }
};

/// Binds an existing constructor taking arguments Args...
template <typename... Args> detail::initimpl::constructor<Args...> init() { return {}; }
/// Like `init<Args...>()`, but the instance is always constructed through the alias class (even
/// when not inheriting on the Python side).
template <typename... Args> detail::initimpl::alias_constructor<Args...> init_alias() { return {}; }

/// Binds a factory function as a constructor
template <typename Func, typename Ret = detail::initimpl::factory<Func>>
Ret init(Func &&f) { return {std::forward<Func>(f)}; }

/// Dual-argument factory function: the first function is called when no alias is needed, the second
/// when an alias is needed (i.e. due to python-side inheritance).  Arguments must be identical.
template <typename CFunc, typename AFunc, typename Ret = detail::initimpl::factory<CFunc, AFunc>>
Ret init(CFunc &&c, AFunc &&a) {
    return {std::forward<CFunc>(c), std::forward<AFunc>(a)};
}

/// Binds pickling functions `__getstate__` and `__setstate__` and ensures that the type
/// returned by `__getstate__` is the same as the argument accepted by `__setstate__`.
template <typename GetState, typename SetState>
detail::initimpl::pickle_factory<GetState, SetState> pickle(GetState &&g, SetState &&s) {
    return {std::forward<GetState>(g), std::forward<SetState>(s)};
}

PYBIND11_NAMESPACE_BEGIN(detail)

inline str enum_name(handle arg) {
    dict entries = arg.get_type().attr("__entries");
    for (auto kv : entries) {
        if (handle(kv.second[int_(0)]).equal(arg))
            return pybind11::str(kv.first);
    }
    return "???";
}

struct enum_base {
    enum_base(handle base, handle parent) : m_base(base), m_parent(parent) { }

    PYBIND11_NOINLINE void init(bool is_arithmetic, bool is_convertible) {
        m_base.attr("__entries") = dict();
        auto property = handle((PyObject *) &PyProperty_Type);
        auto static_property = handle((PyObject *) get_internals().static_property_type);

        m_base.attr("__repr__") = cpp_function(
            [](object arg) -> str {
                handle type = type::handle_of(arg);
                object type_name = type.attr("__name__");
                return pybind11::str("<{}.{}: {}>").format(type_name, enum_name(arg), int_(arg));
            }, name("__repr__"), is_method(m_base)
        );

        m_base.attr("name") = property(cpp_function(&enum_name, name("name"), is_method(m_base)));

        m_base.attr("__str__") = cpp_function(
            [](handle arg) -> str {
                object type_name = type::handle_of(arg).attr("__name__");
                return pybind11::str("{}.{}").format(type_name, enum_name(arg));
            }, name("name"), is_method(m_base)
        );

        m_base.attr("__doc__") = static_property(cpp_function(
            [](handle arg) -> std::string {
                std::string docstring;
                dict entries = arg.attr("__entries");
                if (((PyTypeObject *) arg.ptr())->tp_doc)
                    docstring += std::string(((PyTypeObject *) arg.ptr())->tp_doc) + "\n\n";
                docstring += "Members:";
                for (auto kv : entries) {
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
                for (auto kv : entries)
                    m[kv.first] = kv.second[int_(0)];
                return m;
            }, name("__members__")), none(), none(), ""
        );

        #define PYBIND11_ENUM_OP_STRICT(op, expr, strict_behavior)                     \
            m_base.attr(op) = cpp_function(                                            \
                [](object a, object b) {                                               \
                    if (!type::handle_of(a).is(type::handle_of(b)))                    \
                        strict_behavior;                                               \
                    return expr;                                                       \
                },                                                                     \
                name(op), is_method(m_base), arg("other"))

        #define PYBIND11_ENUM_OP_CONV(op, expr)                                        \
            m_base.attr(op) = cpp_function(                                            \
                [](object a_, object b_) {                                             \
                    int_ a(a_), b(b_);                                                 \
                    return expr;                                                       \
                },                                                                     \
                name(op), is_method(m_base), arg("other"))

        #define PYBIND11_ENUM_OP_CONV_LHS(op, expr)                                    \
            m_base.attr(op) = cpp_function(                                            \
                [](object a_, object b) {                                              \
                    int_ a(a_);                                                        \
                    return expr;                                                       \
                },                                                                     \
                name(op), is_method(m_base), arg("other"))

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

    PYBIND11_NOINLINE void value(char const* name_, object value, const char *doc = nullptr) {
        dict entries = m_base.attr("__entries");
        str name(name_);
        if (entries.contains(name)) {
            std::string type_name = (std::string) str(m_base.attr("__name__"));
            throw value_error(type_name + ": element \"" + std::string(name_) + "\" already exists!");
        }

        entries[name] = std::make_pair(value, doc);
        m_base.attr(name) = value;
    }

    PYBIND11_NOINLINE void export_values() {
        dict entries = m_base.attr("__entries");
        for (auto kv : entries)
            m_parent.attr(kv.first) = kv.second[int_(0)];
    }

    handle m_base;
    handle m_parent;
};

PYBIND11_NAMESPACE_END(detail)

/// Binds C++ enumerations and enumeration classes to Python
template <typename Type> class enum_ : public class_<Type> {
public:
    using Base = class_<Type>;
    using Base::def;
    using Base::attr;
    using Base::def_property_readonly;
    using Base::def_property_readonly_static;
    using Scalar = typename std::underlying_type<Type>::type;

    template <typename... Extra>
    enum_(const handle &scope, const char *name, const Extra&... extra)
      : class_<Type>(scope, name, extra...), m_base(*this, scope) {
        constexpr bool is_arithmetic = detail::any_of<std::is_same<arithmetic, Extra>...>::value;
        constexpr bool is_convertible = std::is_convertible<Type, Scalar>::value;
        m_base.init(is_arithmetic, is_convertible);

        def(init([](Scalar i) { return static_cast<Type>(i); }), arg("value"));
        def_property_readonly("value", [](Type value) { return (Scalar) value; });
        def("__int__", [](Type value) { return (Scalar) value; });
        #if PY_MAJOR_VERSION < 3
            def("__long__", [](Type value) { return (Scalar) value; });
        #endif
        #if PY_MAJOR_VERSION > 3 || (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION >= 8)
            def("__index__", [](Type value) { return (Scalar) value; });
        #endif

        attr("__setstate__") = cpp_function(
            [](detail::value_and_holder &v_h, Scalar arg) {
                detail::initimpl::setstate<Base>(v_h, static_cast<Type>(arg),
                        Py_TYPE(v_h.inst) != v_h.type->type); },
            detail::is_new_style_constructor(),
            pybind11::name("__setstate__"), is_method(*this), arg("state"));
    }

    /// Export enumeration entries into the parent scope
    enum_& export_values() {
        m_base.export_values();
        return *this;
    }

    /// Add an enumeration entry
    enum_& value(char const* name, Type value, const char *doc = nullptr) {
        m_base.value(name, pybind11::cast(value, return_value_policy::copy), doc);
        return *this;
    }

private:
    detail::enum_base m_base;
};

PYBIND11_NAMESPACE_BEGIN(detail)


inline void keep_alive_impl(handle nurse, handle patient) {
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

PYBIND11_NOINLINE inline void keep_alive_impl(size_t Nurse, size_t Patient, function_call &call, handle ret) {
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

inline std::pair<decltype(internals::registered_types_py)::iterator, bool> all_type_info_get_cache(PyTypeObject *type) {
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

template <typename Iterator, typename Sentinel, bool KeyIterator, return_value_policy Policy>
struct iterator_state {
    Iterator it;
    Sentinel end;
    bool first_or_done;
};

PYBIND11_NAMESPACE_END(detail)

/// Makes a python iterator from a first and past-the-end C++ InputIterator.
template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Iterator,
          typename Sentinel,
#ifndef DOXYGEN_SHOULD_SKIP_THIS  // Issue in breathe 4.26.1
          typename ValueType = decltype(*std::declval<Iterator>()),
#endif
          typename... Extra>
iterator make_iterator(Iterator first, Sentinel last, Extra &&... extra) {
    using state = detail::iterator_state<Iterator, Sentinel, false, Policy>;

    if (!detail::get_type_info(typeid(state), false)) {
        class_<state>(handle(), "iterator", pybind11::module_local())
            .def("__iter__", [](state &s) -> state& { return s; })
            .def("__next__", [](state &s) -> ValueType {
                if (!s.first_or_done)
                    ++s.it;
                else
                    s.first_or_done = false;
                if (s.it == s.end) {
                    s.first_or_done = true;
                    throw stop_iteration();
                }
                return *s.it;
            }, std::forward<Extra>(extra)..., Policy);
    }

    return cast(state{first, last, true});
}

/// Makes an python iterator over the keys (`.first`) of a iterator over pairs from a
/// first and past-the-end InputIterator.
template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Iterator,
          typename Sentinel,
#ifndef DOXYGEN_SHOULD_SKIP_THIS  // Issue in breathe 4.26.1
          typename KeyType = decltype((*std::declval<Iterator>()).first),
#endif
          typename... Extra>
iterator make_key_iterator(Iterator first, Sentinel last, Extra &&... extra) {
    using state = detail::iterator_state<Iterator, Sentinel, true, Policy>;

    if (!detail::get_type_info(typeid(state), false)) {
        class_<state>(handle(), "iterator", pybind11::module_local())
            .def("__iter__", [](state &s) -> state& { return s; })
            .def("__next__", [](state &s) -> KeyType {
                if (!s.first_or_done)
                    ++s.it;
                else
                    s.first_or_done = false;
                if (s.it == s.end) {
                    s.first_or_done = true;
                    throw stop_iteration();
                }
                return (*s.it).first;
            }, std::forward<Extra>(extra)..., Policy);
    }

    return cast(state{first, last, true});
}

/// Makes an iterator over values of an stl container or other container supporting
/// `std::begin()`/`std::end()`
template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Type, typename... Extra> iterator make_iterator(Type &value, Extra&&... extra) {
    return make_iterator<Policy>(std::begin(value), std::end(value), extra...);
}

/// Makes an iterator over the keys (`.first`) of a stl map-like container supporting
/// `std::begin()`/`std::end()`
template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Type, typename... Extra> iterator make_key_iterator(Type &value, Extra&&... extra) {
    return make_key_iterator<Policy>(std::begin(value), std::end(value), extra...);
}

template <typename InputType, typename OutputType> void implicitly_convertible() {
    struct set_flag {
        bool &flag;
        set_flag(bool &flag_) : flag(flag_) { flag_ = true; }
        ~set_flag() { flag = false; }
    };
    auto implicit_caster = [](PyObject *obj, PyTypeObject *type) -> PyObject * {
        static bool currently_used = false;
        if (currently_used) // implicit conversions are non-reentrant
            return nullptr;
        set_flag flag_helper(currently_used);
        if (!detail::make_caster<InputType>().load(obj, false))
            return nullptr;
        tuple args(1);
        args[0] = obj;
        PyObject *result = PyObject_Call((PyObject *) type, args.ptr(), nullptr);
        if (result == nullptr)
            PyErr_Clear();
        return result;
    };

    if (auto tinfo = detail::get_type_info(typeid(OutputType)))
        tinfo->implicit_conversions.push_back(implicit_caster);
    else
        pybind11_fail("implicitly_convertible: Unable to find type " + type_id<OutputType>());
}

template <typename ExceptionTranslator>
void register_exception_translator(ExceptionTranslator&& translator) {
    detail::get_internals().registered_exception_translators.push_front(
        std::forward<ExceptionTranslator>(translator));
}

/**
 * Wrapper to generate a new Python exception type.
 *
 * This should only be used with PyErr_SetString for now.
 * It is not (yet) possible to use as a py::base.
 * Template type argument is reserved for future use.
 */
template <typename type>
class exception : public object {
public:
    exception() = default;
    exception(handle scope, const char *name, handle base = PyExc_Exception) {
        std::string full_name = scope.attr("__name__").cast<std::string>() +
                                std::string(".") + name;
        m_ptr = PyErr_NewException(const_cast<char *>(full_name.c_str()), base.ptr(), NULL);
        if (hasattr(scope, "__dict__") && scope.attr("__dict__").contains(name))
            pybind11_fail("Error during initialization: multiple incompatible "
                          "definitions with name \"" + std::string(name) + "\"");
        scope.attr(name) = *this;
    }

    // Sets the current python exception to this exception object with the given message
    void operator()(const char *message) {
        PyErr_SetString(m_ptr, message);
    }
};

PYBIND11_NAMESPACE_BEGIN(detail)
// Returns a reference to a function-local static exception object used in the simple
// register_exception approach below.  (It would be simpler to have the static local variable
// directly in register_exception, but that makes clang <3.5 segfault - issue #1349).
template <typename CppException>
exception<CppException> &get_exception_object() { static exception<CppException> ex; return ex; }
PYBIND11_NAMESPACE_END(detail)

/**
 * Registers a Python exception in `m` of the given `name` and installs an exception translator to
 * translate the C++ exception to the created Python exception using the exceptions what() method.
 * This is intended for simple exception translations; for more complex translation, register the
 * exception object and translator directly.
 */
template <typename CppException>
exception<CppException> &register_exception(handle scope,
                                            const char *name,
                                            handle base = PyExc_Exception) {
    auto &ex = detail::get_exception_object<CppException>();
    if (!ex) ex = exception<CppException>(scope, name, base);

    register_exception_translator([](std::exception_ptr p) {
        if (!p) return;
        try {
            std::rethrow_exception(p);
        } catch (const CppException &e) {
            detail::get_exception_object<CppException>()(e.what());
        }
    });
    return ex;
}

PYBIND11_NAMESPACE_BEGIN(detail)
PYBIND11_NOINLINE inline void print(tuple args, dict kwargs) {
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
            file = module_::import("sys").attr("stdout");
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

template <return_value_policy policy = return_value_policy::automatic_reference, typename... Args>
void print(Args &&...args) {
    auto c = detail::collect_arguments<policy>(std::forward<Args>(args)...);
    detail::print(c.args(), c.kwargs());
}

error_already_set::~error_already_set() {
    if (m_type) {
        gil_scoped_acquire gil;
        error_scope scope;
        m_type.release().dec_ref();
        m_value.release().dec_ref();
        m_trace.release().dec_ref();
    }
}

PYBIND11_NAMESPACE_BEGIN(detail)
inline function get_type_override(const void *this_ptr, const type_info *this_type, const char *name)  {
    handle self = get_object_handle(this_ptr, this_type);
    if (!self)
        return function();
    handle type = type::handle_of(self);
    auto key = std::make_pair(type.ptr(), name);

    /* Cache functions that aren't overridden in Python to avoid
       many costly Python dictionary lookups below */
    auto &cache = get_internals().inactive_override_cache;
    if (cache.find(key) != cache.end())
        return function();

    function override = getattr(self, name, function());
    if (override.is_cpp_function()) {
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

    return override;
}
PYBIND11_NAMESPACE_END(detail)

/** \rst
  Try to retrieve a python method by the provided name from the instance pointed to by the this_ptr.

  :this_ptr: The pointer to the object the overridden method should be retrieved for. This should be
             the first non-trampoline class encountered in the inheritance chain.
  :name: The name of the overridden Python method to retrieve.
  :return: The Python method by this name from the object or an empty function wrapper.
 \endrst */
template <class T> function get_override(const T *this_ptr, const char *name) {
    auto tinfo = detail::get_type_info(typeid(T));
    return tinfo ? detail::get_type_override(this_ptr, tinfo, name) : function();
}

#define PYBIND11_OVERRIDE_IMPL(ret_type, cname, name, ...) \
    do { \
        pybind11::gil_scoped_acquire gil; \
        pybind11::function override = pybind11::get_override(static_cast<const cname *>(this), name); \
        if (override) { \
            auto o = override(__VA_ARGS__); \
            if (pybind11::detail::cast_is_temporary_value_reference<ret_type>::value) { \
                static pybind11::detail::override_caster_t<ret_type> caster; \
                return pybind11::detail::cast_ref<ret_type>(std::move(o), caster); \
            } \
            else return pybind11::detail::cast_safe<ret_type>(std::move(o)); \
        } \
    } while (false)

/** \rst
    Macro to populate the virtual method in the trampoline class. This macro tries to look up a method named 'fn'
    from the Python side, deals with the :ref:`gil` and necessary argument conversions to call this method and return
    the appropriate type. See :ref:`overriding_virtuals` for more information. This macro should be used when the method
    name in C is not the same as the method name in Python. For example with `__str__`.

    .. code-block:: cpp

      std::string toString() override {
        PYBIND11_OVERRIDE_NAME(
            std::string, // Return type (ret_type)
            Animal,      // Parent class (cname)
            "__str__",   // Name of method in Python (name)
            toString,    // Name of function in C++ (fn)
        );
      }
\endrst */
#define PYBIND11_OVERRIDE_NAME(ret_type, cname, name, fn, ...) \
    do { \
        PYBIND11_OVERRIDE_IMPL(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), name, __VA_ARGS__); \
        return cname::fn(__VA_ARGS__); \
    } while (false)

/** \rst
    Macro for pure virtual functions, this function is identical to :c:macro:`PYBIND11_OVERRIDE_NAME`, except that it
    throws if no override can be found.
\endrst */
#define PYBIND11_OVERRIDE_PURE_NAME(ret_type, cname, name, fn, ...) \
    do { \
        PYBIND11_OVERRIDE_IMPL(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), name, __VA_ARGS__); \
        pybind11::pybind11_fail("Tried to call pure virtual function \"" PYBIND11_STRINGIFY(cname) "::" name "\""); \
    } while (false)

/** \rst
    Macro to populate the virtual method in the trampoline class. This macro tries to look up the method
    from the Python side, deals with the :ref:`gil` and necessary argument conversions to call this method and return
    the appropriate type. This macro should be used if the method name in C and in Python are identical.
    See :ref:`overriding_virtuals` for more information.

    .. code-block:: cpp

      class PyAnimal : public Animal {
      public:
          // Inherit the constructors
          using Animal::Animal;

          // Trampoline (need one for each virtual function)
          std::string go(int n_times) override {
              PYBIND11_OVERRIDE_PURE(
                  std::string, // Return type (ret_type)
                  Animal,      // Parent class (cname)
                  go,          // Name of function in C++ (must match Python name) (fn)
                  n_times      // Argument(s) (...)
              );
          }
      };
\endrst */
#define PYBIND11_OVERRIDE(ret_type, cname, fn, ...) \
    PYBIND11_OVERRIDE_NAME(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), #fn, fn, __VA_ARGS__)

/** \rst
    Macro for pure virtual functions, this function is identical to :c:macro:`PYBIND11_OVERRIDE`, except that it throws
    if no override can be found.
\endrst */
#define PYBIND11_OVERRIDE_PURE(ret_type, cname, fn, ...) \
    PYBIND11_OVERRIDE_PURE_NAME(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), #fn, fn, __VA_ARGS__)


// Deprecated versions

PYBIND11_DEPRECATED("get_type_overload has been deprecated")
inline function get_type_overload(const void *this_ptr, const detail::type_info *this_type, const char *name) {
    return detail::get_type_override(this_ptr, this_type, name);
}

template <class T>
inline function get_overload(const T *this_ptr, const char *name) {
    return get_override(this_ptr, name);
}

#define PYBIND11_OVERLOAD_INT(ret_type, cname, name, ...) \
    PYBIND11_OVERRIDE_IMPL(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), name, __VA_ARGS__)
#define PYBIND11_OVERLOAD_NAME(ret_type, cname, name, fn, ...) \
    PYBIND11_OVERRIDE_NAME(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), name, fn, __VA_ARGS__)
#define PYBIND11_OVERLOAD_PURE_NAME(ret_type, cname, name, fn, ...) \
    PYBIND11_OVERRIDE_PURE_NAME(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), name, fn, __VA_ARGS__);
#define PYBIND11_OVERLOAD(ret_type, cname, fn, ...) \
    PYBIND11_OVERRIDE(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), fn, __VA_ARGS__)
#define PYBIND11_OVERLOAD_PURE(ret_type, cname, fn, ...) \
    PYBIND11_OVERRIDE_PURE(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), fn, __VA_ARGS__);

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#  pragma warning(pop)
#elif defined(__GNUG__) && !defined(__clang__)
#  pragma GCC diagnostic pop
#endif
