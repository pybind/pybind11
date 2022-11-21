/*
    pybind11/pybind11.h: Main header file of the C++11 python
    binding generator library

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "detail/class.h"
#include "detail/function_record.h"
#include "detail/init.h"
#include "attr.h"
#include "gil.h"
#include "options.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#if defined(__cpp_lib_launder) && !(defined(_MSC_VER) && (_MSC_VER < 1914))
#    define PYBIND11_STD_LAUNDER std::launder
#    define PYBIND11_HAS_STD_LAUNDER 1
#else
#    define PYBIND11_STD_LAUNDER
#    define PYBIND11_HAS_STD_LAUNDER 0
#endif
#if defined(__GNUG__) && !defined(__clang__)
#    include <cxxabi.h>
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

/* https://stackoverflow.com/questions/46798456/handling-gccs-noexcept-type-warning
   This warning is about ABI compatibility, not code health.
   It is only actually needed in a couple places, but apparently GCC 7 "generates this warning if
   and only if the first template instantiation ... involves noexcept" [stackoverflow], therefore
   it could get triggered from seemingly random places, depending on user code.
   No other GCC version generates this warning.
 */
#if defined(__GNUC__) && __GNUC__ == 7
PYBIND11_WARNING_DISABLE_GCC("-Wnoexcept-type")
#endif

PYBIND11_WARNING_DISABLE_MSVC(4127)

/// Wraps an arbitrary C++ function/method/lambda function/.. into a callable Python object
class cpp_function : public function {
public:
    cpp_function() = default;
    // NOLINTNEXTLINE(google-explicit-constructor)
    cpp_function(std::nullptr_t) {}

    /// Construct a cpp_function from a vanilla function pointer
    template <typename Return, typename... Args, typename... Extra>
    // NOLINTNEXTLINE(google-explicit-constructor)
    cpp_function(Return (*f)(Args...), const Extra &...extra) {
        initialize(f, f, extra...);
    }

    /// Construct a cpp_function from a lambda function (possibly with internal state)
    template <typename Func,
              typename... Extra,
              typename = detail::enable_if_t<detail::is_lambda<Func>::value>>
    // NOLINTNEXTLINE(google-explicit-constructor)
    cpp_function(Func &&f, const Extra &...extra) {
        initialize(
            std::forward<Func>(f), (detail::function_signature_t<Func> *) nullptr, extra...);
    }

    /// Construct a cpp_function from a class method (non-const, no ref-qualifier)
    template <typename Return, typename Class, typename... Arg, typename... Extra>
    // NOLINTNEXTLINE(google-explicit-constructor)
    cpp_function(Return (Class::*f)(Arg...), const Extra &...extra) {
        initialize(
            [f](Class *c, Arg... args) -> Return { return (c->*f)(std::forward<Arg>(args)...); },
            (Return(*)(Class *, Arg...)) nullptr,
            extra...);
    }

    /// Construct a cpp_function from a class method (non-const, lvalue ref-qualifier)
    /// A copy of the overload for non-const functions without explicit ref-qualifier
    /// but with an added `&`.
    template <typename Return, typename Class, typename... Arg, typename... Extra>
    // NOLINTNEXTLINE(google-explicit-constructor)
    cpp_function(Return (Class::*f)(Arg...) &, const Extra &...extra) {
        initialize(
            [f](Class *c, Arg... args) -> Return { return (c->*f)(std::forward<Arg>(args)...); },
            (Return(*)(Class *, Arg...)) nullptr,
            extra...);
    }

    /// Construct a cpp_function from a class method (const, no ref-qualifier)
    template <typename Return, typename Class, typename... Arg, typename... Extra>
    // NOLINTNEXTLINE(google-explicit-constructor)
    cpp_function(Return (Class::*f)(Arg...) const, const Extra &...extra) {
        initialize([f](const Class *c,
                       Arg... args) -> Return { return (c->*f)(std::forward<Arg>(args)...); },
                   (Return(*)(const Class *, Arg...)) nullptr,
                   extra...);
    }

    /// Construct a cpp_function from a class method (const, lvalue ref-qualifier)
    /// A copy of the overload for const functions without explicit ref-qualifier
    /// but with an added `&`.
    template <typename Return, typename Class, typename... Arg, typename... Extra>
    // NOLINTNEXTLINE(google-explicit-constructor)
    cpp_function(Return (Class::*f)(Arg...) const &, const Extra &...extra) {
        initialize([f](const Class *c,
                       Arg... args) -> Return { return (c->*f)(std::forward<Arg>(args)...); },
                   (Return(*)(const Class *, Arg...)) nullptr,
                   extra...);
    }

    /// Return the function name
    object name() const { return attr("__name__"); }

private:
    template <typename Func, typename Return, typename... Args, typename... Extra>
    void initialize(Func &&f, Return (*a)(Args...), const Extra &...extra) {
        constexpr size_t sibling_index = detail::constexpr_first<detail::is_sibling, Extra...>();
        sibling sib = std::get<sibling_index>(std::forward_as_tuple(extra..., sibling(none())));

        bool has_prepend = detail::any_of<std::is_same<prepend, Extra>...>::value;

        handle new_function
            = detail::create_pybind_function_wrapper(std::forward<Func>(f), a, extra...);
        m_ptr = detail::combine_functions(sib.value, new_function, has_prepend).ptr();
    }
};

/// Wrapper for Python extension modules
class module_ : public object {
public:
    PYBIND11_OBJECT_DEFAULT(module_, object, PyModule_Check)

    /// Create a new top-level Python module with the given name and docstring
    PYBIND11_DEPRECATED("Use PYBIND11_MODULE or module_::create_extension_module instead")
    explicit module_(const char *name, const char *doc = nullptr) {
        *this = create_extension_module(name, doc, new PyModuleDef());
    }

    /** \rst
        Create Python binding for a new function within the module scope. ``Func``
        can be a plain C++ function, a function pointer, or a lambda function. For
        details on the ``Extra&& ... extra`` argument, see section :ref:`extras`.
    \endrst */
    template <typename Func, typename... Extra>
    module_ &def(const char *name_, Func &&f, const Extra &...extra) {
        cpp_function func(std::forward<Func>(f),
                          name(name_),
                          scope(*this),
                          sibling(getattr(*this, name_, none())),
                          extra...);
        // NB: allow overwriting here because cpp_function sets up a chain with the intention of
        // overwriting (and has already checked internally that it isn't overwriting
        // non-functions).
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
        const char *this_name = PyModule_GetName(m_ptr);
        if (this_name == nullptr) {
            throw error_already_set();
        }
        std::string full_name = std::string(this_name) + '.' + name;
        handle submodule = PyImport_AddModule(full_name.c_str());
        if (!submodule) {
            throw error_already_set();
        }
        auto result = reinterpret_borrow<module_>(submodule);
        if (doc && options::show_user_defined_docstrings()) {
            result.attr("__doc__") = pybind11::str(doc);
        }
        attr(name) = result;
        return result;
    }

    /// Import and return a module or throws `error_already_set`.
    static module_ import(const char *name) {
        PyObject *obj = PyImport_ImportModule(name);
        if (!obj) {
            throw error_already_set();
        }
        return reinterpret_steal<module_>(obj);
    }

    /// Reload the module or throws `error_already_set`.
    void reload() {
        PyObject *obj = PyImport_ReloadModule(ptr());
        if (!obj) {
            throw error_already_set();
        }
        *this = reinterpret_steal<module_>(obj);
    }

    /** \rst
        Adds an object to the module using the given name.  Throws if an object with the given
    name already exists.

        ``overwrite`` should almost always be false: attempting to overwrite objects that
    pybind11 has established will, in most cases, break things. \endrst */
    PYBIND11_NOINLINE void add_object(const char *name, handle obj, bool overwrite = false) {
        if (!overwrite && hasattr(*this, name)) {
            pybind11_fail(
                "Error during initialization: multiple incompatible definitions with name \""
                + std::string(name) + "\"");
        }

        PyModule_AddObject(ptr(), name, obj.inc_ref().ptr() /* steals a reference */);
    }

    using module_def = PyModuleDef; // TODO: Can this be removed (it was needed only for Python 2)?

    /** \rst
        Create a new top-level module that can be used as the main module of a C extension.

        ``def`` should point to a statically allocated module_def.
    \endrst */
    static module_ create_extension_module(const char *name, const char *doc, module_def *def) {
        // module_def is PyModuleDef
        // Placement new (not an allocation).
        def = new (def) PyModuleDef{
            /* m_base */ PyModuleDef_HEAD_INIT,
            /* m_name */ name,
            /* m_doc */ options::show_user_defined_docstrings() ? doc : nullptr,
            /* m_size */ -1,
            /* m_methods */ nullptr,
            /* m_slots */ nullptr,
            /* m_traverse */ nullptr,
            /* m_clear */ nullptr,
            /* m_free */ nullptr,
        };
        auto *m = PyModule_Create(def);
        if (m == nullptr) {
            if (PyErr_Occurred()) {
                throw error_already_set();
            }
            pybind11_fail("Internal error in module_::create_extension_module()");
        }
        // TODO: Should be reinterpret_steal for Python 3, but Python also steals it again when
        //       returned from PyInit_...
        //       For Python 2, reinterpret_borrow was correct.
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

template <typename... Args, typename = detail::enable_if_t<args_are_all_keyword_or_ds<Args...>()>>
PYBIND11_DEPRECATED("make_simple_namespace should be replaced with "
                    "py::module_::import(\"types\").attr(\"SimpleNamespace\") ")
object make_simple_namespace(Args &&...args_) {
    return module_::import("types").attr("SimpleNamespace")(std::forward<Args>(args_)...);
}

PYBIND11_NAMESPACE_BEGIN(detail)
/// Generic support for creating new Python heap types
class generic_type : public object {
public:
    PYBIND11_OBJECT_DEFAULT(generic_type, object, PyType_Check)
protected:
    void initialize(const type_record &rec) {
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
        if (rec.module_local) {
            get_local_internals().registered_types_cpp[tindex] = tinfo;
        } else {
            internals.registered_types_cpp[tindex] = tinfo;
        }
        internals.registered_types_py[(PyTypeObject *) m_ptr] = {tinfo};

        if (rec.bases.size() > 1 || rec.multiple_inheritance) {
            mark_parents_nonsimple(tinfo->type);
            tinfo->simple_ancestors = false;
        } else if (rec.bases.size() == 1) {
            auto *parent_tinfo = get_type_info((PyTypeObject *) rec.bases[0].ptr());
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

    /// Helper function which tags all parents of a type using mult. inheritance
    void mark_parents_nonsimple(PyTypeObject *value) {
        auto t = reinterpret_borrow<tuple>(value->tp_bases);
        for (handle h : t) {
            auto *tinfo2 = get_type_info((PyTypeObject *) h.ptr());
            if (tinfo2) {
                tinfo2->simple_type = false;
            }
            mark_parents_nonsimple((PyTypeObject *) h.ptr());
        }
    }

    void install_buffer_funcs(buffer_info *(*get_buffer)(PyObject *, void *),
                              void *get_buffer_data) {
        auto *type = (PyHeapTypeObject *) m_ptr;
        auto *tinfo = detail::get_type_info(&type->ht_type);

        if (!type->ht_type.tp_as_buffer) {
            pybind11_fail("To be able to register buffer protocol support for the type '"
                          + get_fully_qualified_tp_name(tinfo->type)
                          + "' the associated class<>(..) invocation must "
                            "include the pybind11::buffer_protocol() annotation!");
        }

        tinfo->get_buffer = get_buffer;
        tinfo->get_buffer_data = get_buffer_data;
    }
};

/// Set the pointer to operator new if it exists. The cast is needed because it can be
/// overloaded.
template <typename T,
          typename = void_t<decltype(static_cast<void *(*) (size_t)>(T::operator new))>>
void set_operator_new(type_record *r) {
    r->operator_new = &T::operator new;
}

template <typename>
void set_operator_new(...) {}

template <typename T, typename SFINAE = void>
struct has_operator_delete : std::false_type {};
template <typename T>
struct has_operator_delete<T, void_t<decltype(static_cast<void (*)(void *)>(T::operator delete))>>
    : std::true_type {};
template <typename T, typename SFINAE = void>
struct has_operator_delete_size : std::false_type {};
template <typename T>
struct has_operator_delete_size<
    T,
    void_t<decltype(static_cast<void (*)(void *, size_t)>(T::operator delete))>> : std::true_type {
};
/// Call class-specific delete if it exists or global otherwise. Can also be an overload set.
template <typename T, enable_if_t<has_operator_delete<T>::value, int> = 0>
void call_operator_delete(T *p, size_t, size_t) {
    T::operator delete(p);
}
template <typename T,
          enable_if_t<!has_operator_delete<T>::value && has_operator_delete_size<T>::value, int>
          = 0>
void call_operator_delete(T *p, size_t s, size_t) {
    T::operator delete(p, s);
}

inline void call_operator_delete(void *p, size_t s, size_t a) {
    (void) s;
    (void) a;
#if defined(__cpp_aligned_new) && (!defined(_MSC_VER) || _MSC_VER >= 1912)
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

inline void add_class_method(object &cls, const char *name_, const cpp_function &cf) {
    cls.attr(cf.name()) = cf;
    if (std::strcmp(name_, "__eq__") == 0 && !cls.attr("__dict__").contains("__hash__")) {
        cls.attr("__hash__") = none();
    }
}

PYBIND11_NAMESPACE_END(detail)

/// Given a pointer to a member function, cast it to its `Derived` version.
/// Forward everything else unchanged.
template <typename /*Derived*/, typename F>
auto method_adaptor(F &&f) -> decltype(std::forward<F>(f)) {
    return std::forward<F>(f);
}

template <typename Derived, typename Return, typename Class, typename... Args>
auto method_adaptor(Return (Class::*pmf)(Args...)) -> Return (Derived::*)(Args...) {
    static_assert(
        detail::is_accessible_base_of<Class, Derived>::value,
        "Cannot bind an inaccessible base class method; use a lambda definition instead");
    return pmf;
}

template <typename Derived, typename Return, typename Class, typename... Args>
auto method_adaptor(Return (Class::*pmf)(Args...) const) -> Return (Derived::*)(Args...) const {
    static_assert(
        detail::is_accessible_base_of<Class, Derived>::value,
        "Cannot bind an inaccessible base class method; use a lambda definition instead");
    return pmf;
}

template <typename type_, typename... options>
class class_ : public detail::generic_type {
    template <typename T>
    using is_holder = detail::is_holder_type<type_, T>;
    template <typename T>
    using is_subtype = detail::is_strict_base_of<type_, T>;
    template <typename T>
    using is_base = detail::is_strict_base_of<T, type_>;
    // struct instead of using here to help MSVC:
    template <typename T>
    struct is_valid_class_option : detail::any_of<is_holder<T>, is_subtype<T>, is_base<T>> {};

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
    class_(handle scope, const char *name, const Extra &...extra) {
        using namespace detail;

        // MI can only be specified via class_ template options, not constructor parameters
        static_assert(
            none_of<is_pyobject<Extra>...>::value || // no base class arguments, or:
                (constexpr_sum(is_pyobject<Extra>::value...) == 1 && // Exactly one base
                 constexpr_sum(is_base<options>::value...) == 0 &&   // no template option bases
                 // no multiple_inheritance attr
                 none_of<std::is_same<multiple_inheritance, Extra>...>::value),
            "Error: multiple inheritance bases must be specified via class_ template options");

        type_record record;
        record.scope = scope;
        record.name = name;
        record.type = &typeid(type);
        record.type_size = sizeof(conditional_t<has_alias, type_alias, type>);
        record.type_align = alignof(conditional_t<has_alias, type_alias, type> &);
        record.holder_size = sizeof(holder_type);
        record.init_instance = init_instance;
        record.dealloc = dealloc;
        record.default_holder = detail::is_instantiation<std::unique_ptr, holder_type>::value;

        set_operator_new<type>(&record);

        /* Register base classes specified via template arguments to class_, if any */
        PYBIND11_EXPAND_SIDE_EFFECTS(add_base<options>(record));

        /* Process optional arguments, if any */
        process_attributes<Extra...>::init(extra..., record);

        generic_type::initialize(record);

        if (has_alias) {
            auto &instances = record.module_local ? get_local_internals().registered_types_cpp
                                                  : get_internals().registered_types_cpp;
            instances[std::type_index(typeid(type_alias))]
                = instances[std::type_index(typeid(type))];
        }
    }

    template <typename Base, detail::enable_if_t<is_base<Base>::value, int> = 0>
    static void add_base(detail::type_record &rec) {
        rec.add_base(typeid(Base), [](void *src) -> void * {
            return static_cast<Base *>(reinterpret_cast<type *>(src));
        });
    }

    template <typename Base, detail::enable_if_t<!is_base<Base>::value, int> = 0>
    static void add_base(detail::type_record &) {}

    template <typename Func, typename... Extra>
    class_ &def(const char *name_, Func &&f, const Extra &...extra) {
        cpp_function cf(method_adaptor<type>(std::forward<Func>(f)),
                        name(name_),
                        is_method(*this),
                        sibling(getattr(*this, name_, none())),
                        extra...);
        add_class_method(*this, name_, cf);
        return *this;
    }

    template <typename Func, typename... Extra>
    class_ &def_static(const char *name_, Func &&f, const Extra &...extra) {
        static_assert(!std::is_member_function_pointer<Func>::value,
                      "def_static(...) called with a non-static member function pointer");
        cpp_function cf(std::forward<Func>(f),
                        name(name_),
                        scope(*this),
                        sibling(getattr(*this, name_, none())),
                        extra...);
        auto cf_name = cf.name();
        attr(std::move(cf_name)) = staticmethod(std::move(cf));
        return *this;
    }

    template <typename T, typename... Extra, detail::enable_if_t<T::op_enable_if_hook, int> = 0>
    class_ &def(const T &op, const Extra &...extra) {
        op.execute(*this, extra...);
        return *this;
    }

    template <typename T, typename... Extra, detail::enable_if_t<T::op_enable_if_hook, int> = 0>
    class_ &def_cast(const T &op, const Extra &...extra) {
        op.execute_cast(*this, extra...);
        return *this;
    }

    template <typename... Args, typename... Extra>
    class_ &def(const detail::initimpl::constructor<Args...> &init, const Extra &...extra) {
        PYBIND11_WORKAROUND_INCORRECT_MSVC_C4100(init);
        init.execute(*this, extra...);
        return *this;
    }

    template <typename... Args, typename... Extra>
    class_ &def(const detail::initimpl::alias_constructor<Args...> &init, const Extra &...extra) {
        PYBIND11_WORKAROUND_INCORRECT_MSVC_C4100(init);
        init.execute(*this, extra...);
        return *this;
    }

    template <typename... Args, typename... Extra>
    class_ &def(detail::initimpl::factory<Args...> &&init, const Extra &...extra) {
        std::move(init).execute(*this, extra...);
        return *this;
    }

    template <typename... Args, typename... Extra>
    class_ &def(detail::initimpl::pickle_factory<Args...> &&pf, const Extra &...extra) {
        std::move(pf).execute(*this, extra...);
        return *this;
    }

    template <typename Func>
    class_ &def_buffer(Func &&func) {
        struct capture {
            Func func;
        };
        auto *ptr = new capture{std::forward<Func>(func)};
        install_buffer_funcs(
            [](PyObject *obj, void *ptr) -> buffer_info * {
                detail::make_caster<type> caster;
                if (!caster.load(obj, false)) {
                    return nullptr;
                }
                return new buffer_info(((capture *) ptr)->func(std::move(caster)));
            },
            ptr);
        weakref(m_ptr, cpp_function([ptr](handle wr) {
                    delete ptr;
                    wr.dec_ref();
                }))
            .release();
        return *this;
    }

    template <typename Return, typename Class, typename... Args>
    class_ &def_buffer(Return (Class::*func)(Args...)) {
        return def_buffer([func](type &obj) { return (obj.*func)(); });
    }

    template <typename Return, typename Class, typename... Args>
    class_ &def_buffer(Return (Class::*func)(Args...) const) {
        return def_buffer([func](const type &obj) { return (obj.*func)(); });
    }

    template <typename C, typename D, typename... Extra>
    class_ &def_readwrite(const char *name, D C::*pm, const Extra &...extra) {
        static_assert(std::is_same<C, type>::value || std::is_base_of<C, type>::value,
                      "def_readwrite() requires a class member (or base class member)");
        auto fget = [pm](const type &c) -> const D & { return c.*pm; };
        auto fset = [pm](type &c, const D &value) { c.*pm = value; };
        def_property(name, std::move(fget), std::move(fset), extra...);
        return *this;
    }

    template <typename C, typename D, typename... Extra>
    class_ &def_readonly(const char *name, const D C::*pm, const Extra &...extra) {
        static_assert(std::is_same<C, type>::value || std::is_base_of<C, type>::value,
                      "def_readonly() requires a class member (or base class member)");
        auto fget = [pm](const type &c) -> const D & { return c.*pm; };
        def_property_readonly(name, std::move(fget), extra...);
        return *this;
    }

    /// Uses return_value_policy::reference_internal by default
    template <typename Getter, typename... Extra>
    class_ &def_property_readonly(const char *name, Getter &&fget, const Extra &...extra) {
        return def_property(name, std::forward<Getter>(fget), nullptr, extra...);
    }

    /// Uses return_value_policy::reference_internal by default
    template <typename Getter, typename Setter, typename... Extra>
    class_ &def_property(const char *name, Getter &&fget, Setter &&fset, const Extra &...extra) {
        return def_property_impl(name,
                                 std::forward<Getter>(fget),
                                 std::forward<Setter>(fset),
                                 is_method(*this),
                                 return_value_policy::reference_internal,
                                 extra...);
    }

    template <typename D, typename... Extra>
    class_ &def_readwrite_static(const char *name, D *pm, const Extra &...extra) {
        auto fget = [pm](const object &) -> const D & { return *pm; };
        auto fset = [pm](const object &, const D &value) { *pm = value; };
        def_property_static(name, std::move(fget), std::move(fset), extra...);
        return *this;
    }

    template <typename D, typename... Extra>
    class_ &def_readonly_static(const char *name, const D *pm, const Extra &...extra) {
        auto fget = [pm](const object &) -> const D & { return *pm; };
        def_property_readonly_static(name, std::move(fget), extra...);
        return *this;
    }

    /// Uses return_value_policy::reference by default
    template <typename Getter, typename... Extra>
    class_ &def_property_readonly_static(const char *name, Getter &&fget, const Extra &...extra) {
        return def_property_static(name, std::forward<Getter>(fget), nullptr, extra...);
    }

    /// Uses return_value_policy::reference by default
    template <typename Getter, typename Setter, typename... Extra>
    class_ &
    def_property_static(const char *name, Getter &&fget, Setter &&fset, const Extra &...extra) {
        return def_property_impl(name,
                                 std::forward<Getter>(fget),
                                 std::forward<Setter>(fset),
                                 scope(*this),
                                 return_value_policy::reference,
                                 extra...);
    }

    template <typename T>
    using convertable_to_handle = detail::any_of<std::is_same<T, std::nullptr_t>,
                                                 std::is_same<T, cpp_function>,
                                                 std::is_same<T, handle>>;

    template <
        typename Getter,
        typename Setter,
        typename... Extra,
        detail::enable_if_t<!convertable_to_handle<detail::remove_cvref_t<Getter>>::value, bool>
        = true>
    class_ &
    def_property_impl(const char *name, Getter &&fget, Setter &&fset, const Extra &...extra) {
        return def_property_impl_1(
            name,
            cpp_function(method_adaptor<type>(std::forward<Getter>(fget)), extra...),
            std::forward<Setter>(fset),
            extra...);
    }

    template <
        typename Getter,
        typename Setter,
        typename... Extra,
        detail::enable_if_t<convertable_to_handle<detail::remove_cvref_t<Getter>>::value, bool>
        = true>
    class_ &
    def_property_impl(const char *name, Getter &&fget, Setter &&fset, const Extra &...extra) {
        return def_property_impl_1(
            name, static_cast<handle>(fget), std::forward<Setter>(fset), extra...);
    }

    template <
        typename Setter,
        typename... Extra,
        detail::enable_if_t<!convertable_to_handle<detail::remove_cvref_t<Setter>>::value, bool>
        = true>
    class_ &
    def_property_impl_1(const char *name, handle fget, Setter &&fset, const Extra &...extra) {
        return def_property_impl_2(
            name,
            fget,
            cpp_function(method_adaptor<type>(std::forward<Setter>(fset)), extra...),
            extra...);
    }

    template <
        typename Setter,
        typename... Extra,
        detail::enable_if_t<convertable_to_handle<detail::remove_cvref_t<Setter>>::value, bool>
        = true>
    class_ &
    def_property_impl_1(const char *name, handle fget, Setter &&fset, const Extra &...extra) {
        return def_property_impl_2(name, fget, static_cast<handle>(fset), extra...);
    }

    template <typename... Extra>
    class_ &def_property_impl_2(const char *name, handle fget, handle fset, const Extra &...) {
        constexpr bool has_is_method = detail::any_of<std::is_same<is_method, Extra>...>::value;
        auto property
            = handle((PyObject *) (has_is_method ? &PyProperty_Type
                                                 : detail::get_internals().static_property_type));
        object documentation = pybind11::str("");
        if (pybind11::options::show_user_defined_docstrings()) {
            if (fset) {
                documentation = getattr(fset, "__doc__");
            }
            if (fget) {
                documentation = getattr(fget, "__doc__");
            }
        }
        attr(name) = property(fget.ptr() ? fget : none(),
                              fset.ptr() ? fset : none(),
                              /*deleter*/ none(),
                              documentation);
        return *this;
    }

private:
    /// Initialize holder object, variant 1: object derives from enable_shared_from_this
    template <typename T>
    static void init_holder(detail::instance *inst,
                            detail::value_and_holder &v_h,
                            const holder_type * /* unused */,
                            const std::enable_shared_from_this<T> * /* dummy */) {

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
                                          const holder_type *holder_ptr,
                                          std::true_type /*is_copy_constructible*/) {
        new (std::addressof(v_h.holder<holder_type>()))
            holder_type(*reinterpret_cast<const holder_type *>(holder_ptr));
    }

    static void init_holder_from_existing(const detail::value_and_holder &v_h,
                                          const holder_type *holder_ptr,
                                          std::false_type /*is_copy_constructible*/) {
        new (std::addressof(v_h.holder<holder_type>()))
            holder_type(std::move(*const_cast<holder_type *>(holder_ptr)));
    }

    /// Initialize holder object, variant 2: try to construct from existing holder object, if
    /// possible
    static void init_holder(detail::instance *inst,
                            detail::value_and_holder &v_h,
                            const holder_type *holder_ptr,
                            const void * /* dummy -- not enable_shared_from_this<T>) */) {
        if (holder_ptr) {
            init_holder_from_existing(v_h, holder_ptr, std::is_copy_constructible<holder_type>());
            v_h.set_holder_constructed();
        } else if (detail::always_construct_holder<holder_type>::value || inst->owned) {
            new (std::addressof(v_h.holder<holder_type>())) holder_type(v_h.value_ptr<type>());
            v_h.set_holder_constructed();
        }
    }

    /// Performs instance initialization including constructing a holder and registering the
    /// known instance.  Should be called as soon as the `type` value_ptr is set for an
    /// instance.  Takes an optional pointer to an existing holder to use; if not specified and
    /// the instance is
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
        } else {
            detail::call_operator_delete(
                v_h.value_ptr<type>(), v_h.type->type_size, v_h.type->type_align);
        }
        v_h.value_ptr() = nullptr;
    }
};

/// Binds an existing constructor taking arguments Args...
template <typename... Args>
detail::initimpl::constructor<Args...> init() {
    return {};
}
/// Like `init<Args...>()`, but the instance is always constructed through the alias class
/// (even when not inheriting on the Python side).
template <typename... Args>
detail::initimpl::alias_constructor<Args...> init_alias() {
    return {};
}

/// Binds a factory function as a constructor
template <typename Func, typename Ret = detail::initimpl::factory<Func>>
Ret init(Func &&f) {
    return {std::forward<Func>(f)};
}

/// Dual-argument factory function: the first function is called when no alias is needed, the
/// second when an alias is needed (i.e. due to python-side inheritance).  Arguments must be
/// identical.
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
        if (handle(kv.second[int_(0)]).equal(arg)) {
            return pybind11::str(kv.first);
        }
    }
    return "???";
}

struct enum_base {
    enum_base(const handle &base, const handle &parent) : m_base(base), m_parent(parent) {}

    PYBIND11_NOINLINE void init(bool is_arithmetic, bool is_convertible) {
        m_base.attr("__entries") = dict();
        auto property = handle((PyObject *) &PyProperty_Type);
        auto static_property = handle((PyObject *) get_internals().static_property_type);

        m_base.attr("__repr__") = cpp_function(
            [](const object &arg) -> str {
                handle type = type::handle_of(arg);
                object type_name = type.attr("__name__");
                return pybind11::str("<{}.{}: {}>")
                    .format(std::move(type_name), enum_name(arg), int_(arg));
            },
            name("__repr__"),
            is_method(m_base));

        m_base.attr("name") = property(cpp_function(&enum_name, name("name"), is_method(m_base)));

        m_base.attr("__str__") = cpp_function(
            [](handle arg) -> str {
                object type_name = type::handle_of(arg).attr("__name__");
                return pybind11::str("{}.{}").format(std::move(type_name), enum_name(arg));
            },
            name("name"),
            is_method(m_base));

        if (options::show_enum_members_docstring()) {
            m_base.attr("__doc__") = static_property(
                cpp_function(
                    [](handle arg) -> std::string {
                        std::string docstring;
                        dict entries = arg.attr("__entries");
                        if (((PyTypeObject *) arg.ptr())->tp_doc) {
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
        arg("other"))

#define PYBIND11_ENUM_OP_CONV(op, expr)                                                           \
    m_base.attr(op) = cpp_function(                                                               \
        [](const object &a_, const object &b_) {                                                  \
            int_ a(a_), b(b_);                                                                    \
            return expr;                                                                          \
        },                                                                                        \
        name(op),                                                                                 \
        is_method(m_base),                                                                        \
        arg("other"))

#define PYBIND11_ENUM_OP_CONV_LHS(op, expr)                                                       \
    m_base.attr(op) = cpp_function(                                                               \
        [](const object &a_, const object &b) {                                                   \
            int_ a(a_);                                                                           \
            return expr;                                                                          \
        },                                                                                        \
        name(op),                                                                                 \
        is_method(m_base),                                                                        \
        arg("other"))

        if (is_convertible) {
            PYBIND11_ENUM_OP_CONV_LHS("__eq__", !b.is_none() && a.equal(b));
            PYBIND11_ENUM_OP_CONV_LHS("__ne__", b.is_none() || !a.equal(b));

            if (is_arithmetic) {
                PYBIND11_ENUM_OP_CONV("__lt__", a < b);
                PYBIND11_ENUM_OP_CONV("__gt__", a > b);
                PYBIND11_ENUM_OP_CONV("__le__", a <= b);
                PYBIND11_ENUM_OP_CONV("__ge__", a >= b);
                PYBIND11_ENUM_OP_CONV("__and__", a & b);
                PYBIND11_ENUM_OP_CONV("__rand__", a & b);
                PYBIND11_ENUM_OP_CONV("__or__", a | b);
                PYBIND11_ENUM_OP_CONV("__ror__", a | b);
                PYBIND11_ENUM_OP_CONV("__xor__", a ^ b);
                PYBIND11_ENUM_OP_CONV("__rxor__", a ^ b);
                m_base.attr("__invert__")
                    = cpp_function([](const object &arg) { return ~(int_(arg)); },
                                   name("__invert__"),
                                   is_method(m_base));
            }
        } else {
            PYBIND11_ENUM_OP_STRICT("__eq__", int_(a).equal(int_(b)), return false);
            PYBIND11_ENUM_OP_STRICT("__ne__", !int_(a).equal(int_(b)), return true);

            if (is_arithmetic) {
#define PYBIND11_THROW throw type_error("Expected an enumeration of matching type!");
                PYBIND11_ENUM_OP_STRICT("__lt__", int_(a) < int_(b), PYBIND11_THROW);
                PYBIND11_ENUM_OP_STRICT("__gt__", int_(a) > int_(b), PYBIND11_THROW);
                PYBIND11_ENUM_OP_STRICT("__le__", int_(a) <= int_(b), PYBIND11_THROW);
                PYBIND11_ENUM_OP_STRICT("__ge__", int_(a) >= int_(b), PYBIND11_THROW);
#undef PYBIND11_THROW
            }
        }

#undef PYBIND11_ENUM_OP_CONV_LHS
#undef PYBIND11_ENUM_OP_CONV
#undef PYBIND11_ENUM_OP_STRICT

        m_base.attr("__getstate__") = cpp_function(
            [](const object &arg) { return int_(arg); }, name("__getstate__"), is_method(m_base));

        m_base.attr("__hash__") = cpp_function(
            [](const object &arg) { return int_(arg); }, name("__hash__"), is_method(m_base));
    }

    PYBIND11_NOINLINE void value(char const *name_, object value, const char *doc = nullptr) {
        dict entries = m_base.attr("__entries");
        str name(name_);
        if (entries.contains(name)) {
            std::string type_name = (std::string) str(m_base.attr("__name__"));
            throw value_error(std::move(type_name) + ": element \"" + std::string(name_)
                              + "\" already exists!");
        }

        entries[name] = pybind11::make_tuple(value, doc);
        m_base.attr(std::move(name)) = std::move(value);
    }

    PYBIND11_NOINLINE void export_values() {
        dict entries = m_base.attr("__entries");
        for (auto kv : entries) {
            m_parent.attr(kv.first) = kv.second[int_(0)];
        }
    }

    handle m_base;
    handle m_parent;
};

template <bool is_signed, size_t length>
struct equivalent_integer {};
template <>
struct equivalent_integer<true, 1> {
    using type = int8_t;
};
template <>
struct equivalent_integer<false, 1> {
    using type = uint8_t;
};
template <>
struct equivalent_integer<true, 2> {
    using type = int16_t;
};
template <>
struct equivalent_integer<false, 2> {
    using type = uint16_t;
};
template <>
struct equivalent_integer<true, 4> {
    using type = int32_t;
};
template <>
struct equivalent_integer<false, 4> {
    using type = uint32_t;
};
template <>
struct equivalent_integer<true, 8> {
    using type = int64_t;
};
template <>
struct equivalent_integer<false, 8> {
    using type = uint64_t;
};

template <typename IntLike>
using equivalent_integer_t =
    typename equivalent_integer<std::is_signed<IntLike>::value, sizeof(IntLike)>::type;

PYBIND11_NAMESPACE_END(detail)

/// Binds C++ enumerations and enumeration classes to Python
template <typename Type>
class enum_ : public class_<Type> {
public:
    using Base = class_<Type>;
    using Base::attr;
    using Base::def;
    using Base::def_property_readonly;
    using Base::def_property_readonly_static;
    using Underlying = typename std::underlying_type<Type>::type;
    // Scalar is the integer representation of underlying type
    using Scalar = detail::conditional_t<detail::any_of<detail::is_std_char_type<Underlying>,
                                                        std::is_same<Underlying, bool>>::value,
                                         detail::equivalent_integer_t<Underlying>,
                                         Underlying>;

    template <typename... Extra>
    enum_(const handle &scope, const char *name, const Extra &...extra)
        : class_<Type>(scope, name, extra...), m_base(*this, scope) {
        constexpr bool is_arithmetic = detail::any_of<std::is_same<arithmetic, Extra>...>::value;
        constexpr bool is_convertible = std::is_convertible<Type, Underlying>::value;
        m_base.init(is_arithmetic, is_convertible);

        def(init([](Scalar i) { return static_cast<Type>(i); }), arg("value"));
        def_property_readonly("value", [](Type value) { return (Scalar) value; });
        def("__int__", [](Type value) { return (Scalar) value; });
        def("__index__", [](Type value) { return (Scalar) value; });
        attr("__setstate__") = cpp_function(
            [](detail::value_and_holder &v_h, Scalar arg) {
                detail::initimpl::setstate<Base>(
                    v_h, static_cast<Type>(arg), Py_TYPE(v_h.inst) != v_h.type->type);
            },
            detail::is_new_style_constructor(),
            pybind11::name("__setstate__"),
            is_method(*this),
            arg("state"));
    }

    /// Export enumeration entries into the parent scope
    enum_ &export_values() {
        m_base.export_values();
        return *this;
    }

    /// Add an enumeration entry
    enum_ &value(char const *name, Type value, const char *doc = nullptr) {
        m_base.value(name, pybind11::cast(value, return_value_policy::copy), doc);
        return *this;
    }

private:
    detail::enum_base m_base;
};

PYBIND11_NAMESPACE_BEGIN(detail)

PYBIND11_NOINLINE void keep_alive_impl(handle nurse, handle patient) {
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

inline std::pair<decltype(internals::registered_types_py)::iterator, bool>
all_type_info_get_cache(PyTypeObject *type) {
    auto res = get_internals()
                   .registered_types_py
#ifdef __cpp_lib_unordered_map_try_emplace
                   .try_emplace(type);
#else
                   .emplace(type, std::vector<detail::type_info *>());
#endif
    if (res.second) {
        // New cache entry created; set up a weak reference to automatically remove it if the
        // type gets destroyed:
        weakref((PyObject *) type, cpp_function([type](handle wr) {
                    get_internals().registered_types_py.erase(type);

                    // TODO consolidate the erasure code in pybind11_meta_dealloc() in class.h
                    auto &cache = get_internals().inactive_override_cache;
                    for (auto it = cache.begin(), last = cache.end(); it != last;) {
                        if (it->first == reinterpret_cast<PyObject *>(type)) {
                            it = cache.erase(it);
                        } else {
                            ++it;
                        }
                    }

                    wr.dec_ref();
                }))
            .release();
    }

    return res;
}

/* There are a large number of apparently unused template arguments because
 * each combination requires a separate py::class_ registration.
 */
template <typename Access,
          return_value_policy Policy,
          typename Iterator,
          typename Sentinel,
          typename ValueType,
          typename... Extra>
struct iterator_state {
    Iterator it;
    Sentinel end;
    bool first_or_done;
};

// Note: these helpers take the iterator by non-const reference because some
// iterators in the wild can't be dereferenced when const. The & after Iterator
// is required for MSVC < 16.9. SFINAE cannot be reused for result_type due to
// bugs in ICC, NVCC, and PGI compilers. See PR #3293.
template <typename Iterator, typename SFINAE = decltype(*std::declval<Iterator &>())>
struct iterator_access {
    using result_type = decltype(*std::declval<Iterator &>());
    // NOLINTNEXTLINE(readability-const-return-type) // PR #3263
    result_type operator()(Iterator &it) const { return *it; }
};

template <typename Iterator, typename SFINAE = decltype((*std::declval<Iterator &>()).first)>
class iterator_key_access {
private:
    using pair_type = decltype(*std::declval<Iterator &>());

public:
    /* If either the pair itself or the element of the pair is a reference, we
     * want to return a reference, otherwise a value. When the decltype
     * expression is parenthesized it is based on the value category of the
     * expression; otherwise it is the declared type of the pair member.
     * The use of declval<pair_type> in the second branch rather than directly
     * using *std::declval<Iterator &>() is a workaround for nvcc
     * (it's not used in the first branch because going via decltype and back
     * through declval does not perfectly preserve references).
     */
    using result_type
        = conditional_t<std::is_reference<decltype(*std::declval<Iterator &>())>::value,
                        decltype(((*std::declval<Iterator &>()).first)),
                        decltype(std::declval<pair_type>().first)>;
    result_type operator()(Iterator &it) const { return (*it).first; }
};

template <typename Iterator, typename SFINAE = decltype((*std::declval<Iterator &>()).second)>
class iterator_value_access {
private:
    using pair_type = decltype(*std::declval<Iterator &>());

public:
    using result_type
        = conditional_t<std::is_reference<decltype(*std::declval<Iterator &>())>::value,
                        decltype(((*std::declval<Iterator &>()).second)),
                        decltype(std::declval<pair_type>().second)>;
    result_type operator()(Iterator &it) const { return (*it).second; }
};

template <typename Access,
          return_value_policy Policy,
          typename Iterator,
          typename Sentinel,
          typename ValueType,
          typename... Extra>
iterator make_iterator_impl(Iterator first, Sentinel last, Extra &&...extra) {
    using state = detail::iterator_state<Access, Policy, Iterator, Sentinel, ValueType, Extra...>;
    // TODO: state captures only the types of Extra, not the values

    if (!detail::get_type_info(typeid(state), false)) {
        class_<state>(handle(), "iterator", pybind11::module_local())
            .def("__iter__", [](state &s) -> state & { return s; })
            .def(
                "__next__",
                [](state &s) -> ValueType {
                    if (!s.first_or_done) {
                        ++s.it;
                    } else {
                        s.first_or_done = false;
                    }
                    if (s.it == s.end) {
                        s.first_or_done = true;
                        throw stop_iteration();
                    }
                    return Access()(s.it);
                    // NOLINTNEXTLINE(readability-const-return-type) // PR #3263
                },
                std::forward<Extra>(extra)...,
                Policy);
    }

    return cast(state{first, last, true});
}

PYBIND11_NAMESPACE_END(detail)

/// Makes a python iterator from a first and past-the-end C++ InputIterator.
template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Iterator,
          typename Sentinel,
          typename ValueType = typename detail::iterator_access<Iterator>::result_type,
          typename... Extra>
iterator make_iterator(Iterator first, Sentinel last, Extra &&...extra) {
    return detail::make_iterator_impl<detail::iterator_access<Iterator>,
                                      Policy,
                                      Iterator,
                                      Sentinel,
                                      ValueType,
                                      Extra...>(first, last, std::forward<Extra>(extra)...);
}

/// Makes a python iterator over the keys (`.first`) of a iterator over pairs from a
/// first and past-the-end InputIterator.
template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Iterator,
          typename Sentinel,
          typename KeyType = typename detail::iterator_key_access<Iterator>::result_type,
          typename... Extra>
iterator make_key_iterator(Iterator first, Sentinel last, Extra &&...extra) {
    return detail::make_iterator_impl<detail::iterator_key_access<Iterator>,
                                      Policy,
                                      Iterator,
                                      Sentinel,
                                      KeyType,
                                      Extra...>(first, last, std::forward<Extra>(extra)...);
}

/// Makes a python iterator over the values (`.second`) of a iterator over pairs from a
/// first and past-the-end InputIterator.
template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Iterator,
          typename Sentinel,
          typename ValueType = typename detail::iterator_value_access<Iterator>::result_type,
          typename... Extra>
iterator make_value_iterator(Iterator first, Sentinel last, Extra &&...extra) {
    return detail::make_iterator_impl<detail::iterator_value_access<Iterator>,
                                      Policy,
                                      Iterator,
                                      Sentinel,
                                      ValueType,
                                      Extra...>(first, last, std::forward<Extra>(extra)...);
}

/// Makes an iterator over values of an stl container or other container supporting
/// `std::begin()`/`std::end()`
template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Type,
          typename... Extra>
iterator make_iterator(Type &value, Extra &&...extra) {
    return make_iterator<Policy>(
        std::begin(value), std::end(value), std::forward<Extra>(extra)...);
}

/// Makes an iterator over the keys (`.first`) of a stl map-like container supporting
/// `std::begin()`/`std::end()`
template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Type,
          typename... Extra>
iterator make_key_iterator(Type &value, Extra &&...extra) {
    return make_key_iterator<Policy>(
        std::begin(value), std::end(value), std::forward<Extra>(extra)...);
}

/// Makes an iterator over the values (`.second`) of a stl map-like container supporting
/// `std::begin()`/`std::end()`
template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Type,
          typename... Extra>
iterator make_value_iterator(Type &value, Extra &&...extra) {
    return make_value_iterator<Policy>(
        std::begin(value), std::end(value), std::forward<Extra>(extra)...);
}

template <typename InputType, typename OutputType>
void implicitly_convertible() {
    struct set_flag {
        bool &flag;
        explicit set_flag(bool &flag_) : flag(flag_) { flag_ = true; }
        ~set_flag() { flag = false; }
    };
    auto implicit_caster = [](PyObject *obj, PyTypeObject *type) -> PyObject * {
        static bool currently_used = false;
        if (currently_used) { // implicit conversions are non-reentrant
            return nullptr;
        }
        set_flag flag_helper(currently_used);
        if (!detail::make_caster<InputType>().load(obj, false)) {
            return nullptr;
        }
        tuple args(1);
        args[0] = obj;
        PyObject *result = PyObject_Call((PyObject *) type, args.ptr(), nullptr);
        if (result == nullptr) {
            PyErr_Clear();
        }
        return result;
    };

    if (auto *tinfo = detail::get_type_info(typeid(OutputType))) {
        tinfo->implicit_conversions.emplace_back(std::move(implicit_caster));
    } else {
        pybind11_fail("implicitly_convertible: Unable to find type " + type_id<OutputType>());
    }
}

inline void register_exception_translator(ExceptionTranslator &&translator) {
    detail::get_internals().registered_exception_translators.push_front(
        std::forward<ExceptionTranslator>(translator));
}

/**
 * Add a new module-local exception translator. Locally registered functions
 * will be tried before any globally registered exception translators, which
 * will only be invoked if the module-local handlers do not deal with
 * the exception.
 */
inline void register_local_exception_translator(ExceptionTranslator &&translator) {
    detail::get_local_internals().registered_exception_translators.push_front(
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
        std::string full_name
            = scope.attr("__name__").cast<std::string>() + std::string(".") + name;
        m_ptr = PyErr_NewException(const_cast<char *>(full_name.c_str()), base.ptr(), nullptr);
        if (hasattr(scope, "__dict__") && scope.attr("__dict__").contains(name)) {
            pybind11_fail("Error during initialization: multiple incompatible "
                          "definitions with name \""
                          + std::string(name) + "\"");
        }
        scope.attr(name) = *this;
    }

    // Sets the current python exception to this exception object with the given message
    void operator()(const char *message) { PyErr_SetString(m_ptr, message); }
};

PYBIND11_NAMESPACE_BEGIN(detail)
// Returns a reference to a function-local static exception object used in the simple
// register_exception approach below.  (It would be simpler to have the static local variable
// directly in register_exception, but that makes clang <3.5 segfault - issue #1349).
template <typename CppException>
exception<CppException> &get_exception_object() {
    static exception<CppException> ex;
    return ex;
}

// Helper function for register_exception and register_local_exception
template <typename CppException>
exception<CppException> &
register_exception_impl(handle scope, const char *name, handle base, bool isLocal) {
    auto &ex = detail::get_exception_object<CppException>();
    if (!ex) {
        ex = exception<CppException>(scope, name, base);
    }

    auto register_func
        = isLocal ? &register_local_exception_translator : &register_exception_translator;

    register_func([](std::exception_ptr p) {
        if (!p) {
            return;
        }
        try {
            std::rethrow_exception(p);
        } catch (const CppException &e) {
            detail::get_exception_object<CppException>()(e.what());
        }
    });
    return ex;
}

// Apply all the extensions translators from a list
// Return true if one of the translators completed without raising an exception
// itself. Return of false indicates that if there are other translators
// available, they should be tried.
inline bool apply_exception_translators(std::forward_list<ExceptionTranslator> &translators) {
    auto last_exception = std::current_exception();

    for (auto &translator : translators) {
        try {
            translator(last_exception);
            return true;
        } catch (...) {
            last_exception = std::current_exception();
        }
    }
    return false;
}

template <typename F>
inline PyObject *handle_exception(const F &f) {
    try {
        return f();
    } catch (error_already_set &e) {
        e.restore();
        return nullptr;
#ifdef __GLIBCXX__
    } catch (abi::__forced_unwind &) {
        throw;
#endif
    } catch (...) {
        /* When an exception is caught, give each registered exception
        translator a chance to translate it to a Python exception. First
        all module-local translators will be tried in reverse order of
        registration. If none of the module-locale translators handle
        the exception (or there are no module-locale translators) then
        the global translators will be tried, also in reverse order of
        registration.
        A translator may choose to do one of the following:
            - catch the exception and call PyErr_SetString or PyErr_SetObject
            to set a standard (or custom) Python exception, or
            - do nothing and let the exception fall through to the next translator, or
            - delegate translation to the next translator by throwing a new type of
        exception.
        */

        auto &local_exception_translators = get_local_internals().registered_exception_translators;
        if (apply_exception_translators(local_exception_translators)) {
            return nullptr;
        }
        auto &exception_translators = get_internals().registered_exception_translators;
        if (apply_exception_translators(exception_translators)) {
            return nullptr;
        }

        PyErr_SetString(PyExc_SystemError, "Exception escaped from default exception translator!");
        return nullptr;
    }
}

PYBIND11_NAMESPACE_END(detail)

/**
 * Registers a Python exception in `m` of the given `name` and installs a translator to
 * translate the C++ exception to the created Python exception using the what() method.
 * This is intended for simple exception translations; for more complex translation, register
 * the exception object and translator directly.
 */
template <typename CppException>
exception<CppException> &
register_exception(handle scope, const char *name, handle base = PyExc_Exception) {
    return detail::register_exception_impl<CppException>(scope, name, base, false /* isLocal */);
}

/**
 * Registers a Python exception in `m` of the given `name` and installs a translator to
 * translate the C++ exception to the created Python exception using the what() method.
 * This translator will only be used for exceptions that are thrown in this module and will be
 * tried before global exception translators, including those registered with register_exception.
 * This is intended for simple exception translations; for more complex translation, register the
 * exception object and translator directly.
 */
template <typename CppException>
exception<CppException> &
register_local_exception(handle scope, const char *name, handle base = PyExc_Exception) {
    return detail::register_exception_impl<CppException>(scope, name, base, true /* isLocal */);
}

PYBIND11_NAMESPACE_BEGIN(detail)
PYBIND11_NOINLINE void print(const tuple &args, const dict &kwargs) {
    auto strings = tuple(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
        strings[i] = str(args[i]);
    }
    auto sep = kwargs.contains("sep") ? kwargs["sep"] : str(" ");
    auto line = sep.attr("join")(std::move(strings));

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
    write(std::move(line));
    write(kwargs.contains("end") ? kwargs["end"] : str("\n"));

    if (kwargs.contains("flush") && kwargs["flush"].cast<bool>()) {
        file.attr("flush")();
    }
}
PYBIND11_NAMESPACE_END(detail)

template <return_value_policy policy = return_value_policy::automatic_reference, typename... Args>
void print(Args &&...args) {
    auto c = detail::collect_arguments<policy>(std::forward<Args>(args)...);
    detail::print(c.args(), c.kwargs());
}

inline void
error_already_set::m_fetched_error_deleter(detail::error_fetch_and_normalize *raw_ptr) {
    gil_scoped_acquire gil;
    error_scope scope;
    delete raw_ptr;
}

inline const char *error_already_set::what() const noexcept {
    gil_scoped_acquire gil;
    error_scope scope;
    return m_fetched_error->error_string().c_str();
}

PYBIND11_NAMESPACE_BEGIN(detail)

inline bool is_pybind11_function(const function &func) {
    PyObject *ptr = func.ptr();
    if (ptr == nullptr) {
        return false;
    }

    if (PyMethod_Check(ptr)) {
        ptr = PyMethod_GET_FUNCTION(ptr);
        if (ptr == nullptr) {
            return false;
        }
    }

    return PyObject_TypeCheck(ptr, pybind_function_type());
}

inline function
get_type_override(const void *this_ptr, const type_info *this_type, const char *name) {
    handle self = get_object_handle(this_ptr, this_type);
    if (!self) {
        return function();
    }
    handle type = type::handle_of(self);
    auto key = std::make_pair(type.ptr(), name);

    /* Cache functions that aren't overridden in Python to avoid
       many costly Python dictionary lookups below */
    auto &cache = get_internals().inactive_override_cache;
    if (cache.find(key) != cache.end()) {
        return function();
    }

    function override = getattr(self, name, function());

    if (is_pybind11_function(override)) {
        // TODO: Fix this
        cache.insert(std::move(key));
        return function();
    }

/* Don't call dispatch code if invoked from overridden function.
   Unfortunately this doesn't work on PyPy. */
#if !defined(PYPY_VERSION)
#    if PY_VERSION_HEX >= 0x03090000
    PyFrameObject *frame = PyThreadState_GetFrame(PyThreadState_Get());
    if (frame != nullptr) {
        PyCodeObject *f_code = PyFrame_GetCode(frame);
        // f_code is guaranteed to not be NULL
        if ((std::string) str(f_code->co_name) == name && f_code->co_argcount > 0) {
            PyObject *locals = PyEval_GetLocals();
            if (locals != nullptr) {
                PyObject *co_varnames = PyObject_GetAttrString((PyObject *) f_code, "co_varnames");
                PyObject *self_arg = PyTuple_GET_ITEM(co_varnames, 0);
                Py_DECREF(co_varnames);
                PyObject *self_caller = dict_getitem(locals, self_arg);
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
#    else
    PyFrameObject *frame = PyThreadState_Get()->frame;
    if (frame != nullptr && (std::string) str(frame->f_code->co_name) == name
        && frame->f_code->co_argcount > 0) {
        PyFrame_FastToLocals(frame);
        PyObject *self_caller
            = dict_getitem(frame->f_locals, PyTuple_GET_ITEM(frame->f_code->co_varnames, 0));
        if (self_caller == self.ptr()) {
            return function();
        }
    }
#    endif

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

/** \rst
  Try to retrieve a python method by the provided name from the instance pointed to by the
  this_ptr.

  :this_ptr: The pointer to the object the overridden method should be retrieved for. This should
             be the first non-trampoline class encountered in the inheritance chain.
  :name: The name of the overridden Python method to retrieve.
  :return: The Python method by this name from the object or an empty function wrapper.
 \endrst */
template <class T>
function get_override(const T *this_ptr, const char *name) {
    auto *tinfo = detail::get_type_info(typeid(T));
    return tinfo ? detail::get_type_override(this_ptr, tinfo, name) : function();
}

#define PYBIND11_OVERRIDE_IMPL(ret_type, cname, name, ...)                                        \
    do {                                                                                          \
        pybind11::gil_scoped_acquire gil;                                                         \
        pybind11::function override                                                               \
            = pybind11::get_override(static_cast<const cname *>(this), name);                     \
        if (override) {                                                                           \
            auto o = override(__VA_ARGS__);                                                       \
            if (pybind11::detail::cast_is_temporary_value_reference<ret_type>::value) {           \
                static pybind11::detail::override_caster_t<ret_type> caster;                      \
                return pybind11::detail::cast_ref<ret_type>(std::move(o), caster);                \
            }                                                                                     \
            return pybind11::detail::cast_safe<ret_type>(std::move(o));                           \
        }                                                                                         \
    } while (false)

/** \rst
    Macro to populate the virtual method in the trampoline class. This macro tries to look up a
    method named 'fn' from the Python side, deals with the :ref:`gil` and necessary argument
    conversions to call this method and return the appropriate type.
    See :ref:`overriding_virtuals` for more information. This macro should be used when the method
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
#define PYBIND11_OVERRIDE_NAME(ret_type, cname, name, fn, ...)                                    \
    do {                                                                                          \
        PYBIND11_OVERRIDE_IMPL(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), name, __VA_ARGS__); \
        return cname::fn(__VA_ARGS__);                                                            \
    } while (false)

/** \rst
    Macro for pure virtual functions, this function is identical to
    :c:macro:`PYBIND11_OVERRIDE_NAME`, except that it throws if no override can be found.
\endrst */
#define PYBIND11_OVERRIDE_PURE_NAME(ret_type, cname, name, fn, ...)                               \
    do {                                                                                          \
        PYBIND11_OVERRIDE_IMPL(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), name, __VA_ARGS__); \
        pybind11::pybind11_fail(                                                                  \
            "Tried to call pure virtual function \"" PYBIND11_STRINGIFY(cname) "::" name "\"");   \
    } while (false)

/** \rst
    Macro to populate the virtual method in the trampoline class. This macro tries to look up the
    method from the Python side, deals with the :ref:`gil` and necessary argument conversions to
    call this method and return the appropriate type. This macro should be used if the method name
    in C and in Python are identical.
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
#define PYBIND11_OVERRIDE(ret_type, cname, fn, ...)                                               \
    PYBIND11_OVERRIDE_NAME(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), #fn, fn, __VA_ARGS__)

/** \rst
    Macro for pure virtual functions, this function is identical to :c:macro:`PYBIND11_OVERRIDE`,
    except that it throws if no override can be found.
\endrst */
#define PYBIND11_OVERRIDE_PURE(ret_type, cname, fn, ...)                                          \
    PYBIND11_OVERRIDE_PURE_NAME(                                                                  \
        PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), #fn, fn, __VA_ARGS__)

// Deprecated versions

PYBIND11_DEPRECATED("get_type_overload has been deprecated")
inline function
get_type_overload(const void *this_ptr, const detail::type_info *this_type, const char *name) {
    return detail::get_type_override(this_ptr, this_type, name);
}

template <class T>
inline function get_overload(const T *this_ptr, const char *name) {
    return get_override(this_ptr, name);
}

#define PYBIND11_OVERLOAD_INT(ret_type, cname, name, ...)                                         \
    PYBIND11_OVERRIDE_IMPL(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), name, __VA_ARGS__)
#define PYBIND11_OVERLOAD_NAME(ret_type, cname, name, fn, ...)                                    \
    PYBIND11_OVERRIDE_NAME(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), name, fn, __VA_ARGS__)
#define PYBIND11_OVERLOAD_PURE_NAME(ret_type, cname, name, fn, ...)                               \
    PYBIND11_OVERRIDE_PURE_NAME(                                                                  \
        PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), name, fn, __VA_ARGS__);
#define PYBIND11_OVERLOAD(ret_type, cname, fn, ...)                                               \
    PYBIND11_OVERRIDE(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), fn, __VA_ARGS__)
#define PYBIND11_OVERLOAD_PURE(ret_type, cname, fn, ...)                                          \
    PYBIND11_OVERRIDE_PURE(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), fn, __VA_ARGS__);

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
