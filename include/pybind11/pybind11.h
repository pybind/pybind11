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
#include "options.h"
#include "detail/class.h"
#include "detail/init.h"

#if defined(__GNUG__) && !defined(__clang__)
#  include <cxxabi.h>
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

/// Wraps an arbitrary C++ function/method/lambda function/.. into a callable Python object
class cpp_function : public function {
public:
    cpp_function();
    cpp_function(std::nullptr_t);

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
    object name() const;

protected:
    PYBIND11_NOINLINE detail::function_record *make_function_record();

    /// Special internal constructor for functors, lambda functions, etc.
    template <typename Func, typename Return, typename... Args, typename... Extra>
    void initialize(Func &&f, Return (*)(Args...), const Extra&... extra) {
        using namespace detail;
        struct capture { remove_reference_t<Func> f; };

        /* Store the function including any extra state it might have (e.g. a lambda capture object) */
        auto rec = make_function_record();

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
            capture *cap = const_cast<capture *>(reinterpret_cast<const capture *>(data));

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
            constexpr bool has_kwonly_args = any_of<std::is_same<kwonly, Extra>...>::value,
                           has_args = any_of<std::is_same<args, Args>...>::value,
                           has_arg_annotations = any_of<is_keyword<Extra>...>::value;
            static_assert(has_arg_annotations || !has_kwonly_args, "py::kwonly requires the use of argument annotations");
            static_assert(!(has_args && has_kwonly_args), "py::kwonly cannot be combined with a py::args argument");
        }

        /* Generate a readable signature describing the function's arguments and return value types */
        static constexpr auto signature = _("(") + cast_in::arg_names + _(") -> ") + cast_out::name;
        PYBIND11_DESCR_CONSTEXPR auto types = decltype(signature)::types();

        /* Register the function with Python from generic (non-templated) code */
        initialize_generic(rec, signature.text, types.data(), sizeof...(Args));

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

    /// Register a function call with Python (generic non-templated code goes here)
    void initialize_generic(detail::function_record *rec, const char *text,
                            const std::type_info *const *types, size_t args);

    /// When a cpp_function is GCed, release any memory allocated by pybind11
    static void destruct(detail::function_record *rec);

    /// Main dispatch logic for calls to functions bound using pybind11
    static PyObject *dispatcher(PyObject *self, PyObject *args_in, PyObject *kwargs_in);
};

/// Wrapper for Python extension modules
class module : public object {
public:
    PYBIND11_OBJECT_DEFAULT(module, object, PyModule_Check)

    /// Create a new top-level Python module with the given name and docstring
    explicit module(const char *name, const char *doc = nullptr);

    /** \rst
        Create Python binding for a new function within the module scope. ``Func``
        can be a plain C++ function, a function pointer, or a lambda function. For
        details on the ``Extra&& ... extra`` argument, see section :ref:`extras`.
    \endrst */
    template <typename Func, typename... Extra>
    module &def(const char *name_, Func &&f, const Extra& ... extra) {
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

            py::module m("example", "pybind11 example plugin");
            py::module m2 = m.def_submodule("sub", "A submodule of 'example'");
            py::module m3 = m2.def_submodule("subsub", "A submodule of 'example.sub'");
    \endrst */
    module def_submodule(const char *name, const char *doc = nullptr);

    /// Import and return a module or throws `error_already_set`.
    static module import(const char *name);

    /// Reload the module or throws `error_already_set`.
    void reload();

    // Adds an object to the module using the given name.  Throws if an object with the given name
    // already exists.
    //
    // overwrite should almost always be false: attempting to overwrite objects that pybind11 has
    // established will, in most cases, break things.
    PYBIND11_NOINLINE void add_object(const char *name, handle obj, bool overwrite = false);
};

/// \ingroup python_builtins
/// Return a dictionary representing the global variables in the current execution frame,
/// or ``__main__.__dict__`` if there is no frame (usually when the interpreter is embedded).
PYBIND11_INLINE dict globals();

PYBIND11_NAMESPACE_BEGIN(detail)
/// Generic support for creating new Python heap types
class generic_type : public object {
    template <typename...> friend class class_;
public:
    PYBIND11_OBJECT_DEFAULT(generic_type, object, PyType_Check)
protected:
    void initialize(const type_record &rec);

    /// Helper function which tags all parents of a type using mult. inheritance
    void mark_parents_nonsimple(PyTypeObject *value);

    void install_buffer_funcs(
            buffer_info *(*get_buffer)(PyObject *, void *),
            void *get_buffer_data);

    // rec_func must be set for either fget or fset.
    void def_property_static_impl(const char *name,
                                  handle fget, handle fset,
                                  detail::function_record *rec_func);
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

PYBIND11_INLINE void call_operator_delete(void *p, size_t s, size_t a);

PYBIND11_INLINE void add_class_method(object& cls, const char *name_, const cpp_function &cf);

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

    template <typename Func> class_& def_buffer(Func &&func) {
        struct capture { Func func; };
        capture *ptr = new capture { std::forward<Func>(func) };
        install_buffer_funcs([](PyObject *obj, void *ptr) -> buffer_info* {
            detail::make_caster<type> caster;
            if (!caster.load(obj, false))
                return nullptr;
            return new buffer_info(((capture *) ptr)->func(caster));
        }, ptr);
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
        try {
            auto sh = std::dynamic_pointer_cast<typename holder_type::element_type>(
                    v_h.value_ptr<type>()->shared_from_this());
            if (sh) {
                new (std::addressof(v_h.holder<holder_type>())) holder_type(std::move(sh));
                v_h.set_holder_constructed();
            }
        } catch (const std::bad_weak_ptr &) {}

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
struct enum_base {
    enum_base(handle base, handle parent);

    PYBIND11_NOINLINE void init(bool is_arithmetic, bool is_convertible);

    PYBIND11_NOINLINE void value(char const* name_, object value, const char *doc = nullptr);

    PYBIND11_NOINLINE void export_values();

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

        def(init([](Scalar i) { return static_cast<Type>(i); }));
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
            pybind11::name("__setstate__"), is_method(*this));
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


void keep_alive_impl(handle nurse, handle patient);

PYBIND11_NOINLINE void keep_alive_impl(size_t Nurse, size_t Patient, function_call &call, handle ret);

std::pair<decltype(internals::registered_types_py)::iterator, bool> all_type_info_get_cache(PyTypeObject *type);

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
          typename ValueType = decltype(*std::declval<Iterator>()),
          typename... Extra>
iterator make_iterator(Iterator first, Sentinel last, Extra &&... extra) {
    typedef detail::iterator_state<Iterator, Sentinel, false, Policy> state;

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
          typename KeyType = decltype((*std::declval<Iterator>()).first),
          typename... Extra>
iterator make_key_iterator(Iterator first, Sentinel last, Extra &&... extra) {
    typedef detail::iterator_state<Iterator, Sentinel, true, Policy> state;

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
        set_flag(bool &flag) : flag(flag) { flag = true; }
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
    exception(handle scope, const char *name, PyObject *base = PyExc_Exception) {
        std::string full_name = scope.attr("__name__").cast<std::string>() +
                                std::string(".") + name;
        m_ptr = PyErr_NewException(const_cast<char *>(full_name.c_str()), base, NULL);
        if (hasattr(scope, name))
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
                                            PyObject *base = PyExc_Exception) {
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
PYBIND11_NOINLINE void print(tuple args, dict kwargs);
PYBIND11_NAMESPACE_END(detail)

template <return_value_policy policy = return_value_policy::automatic_reference, typename... Args>
void print(Args &&...args) {
    auto c = detail::collect_arguments<policy>(std::forward<Args>(args)...);
    detail::print(c.args(), c.kwargs());
}

#if defined(WITH_THREAD) && !defined(PYPY_VERSION)

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
    PYBIND11_NOINLINE gil_scoped_acquire();

    void inc_ref();

    PYBIND11_NOINLINE void dec_ref();

    PYBIND11_NOINLINE ~gil_scoped_acquire();
private:
    PyThreadState *tstate = nullptr;
    bool release = true;
};

class gil_scoped_release {
public:
    explicit gil_scoped_release(bool disassoc = false);
    ~gil_scoped_release();
private:
    PyThreadState *tstate;
    bool disassoc;
};
#elif defined(PYPY_VERSION)
class gil_scoped_acquire {
    PyGILState_STATE state;
public:
    gil_scoped_acquire();
    ~gil_scoped_acquire();
};

class gil_scoped_release {
    PyThreadState *state;
public:
    gil_scoped_release();
    ~gil_scoped_release();
};
#else
class gil_scoped_acquire { };
class gil_scoped_release { };
#endif

PYBIND11_INLINE function get_type_overload(const void *this_ptr, const detail::type_info *this_type, const char *name) ;

/** \rst
  Try to retrieve a python method by the provided name from the instance pointed to by the this_ptr.

  :this_ptr: The pointer to the object the overload should be retrieved for. This should be the first
                   non-trampoline class encountered in the inheritance chain.
  :name: The name of the overloaded Python method to retrieve.
  :return: The Python method by this name from the object or an empty function wrapper.
 \endrst */
template <class T> function get_overload(const T *this_ptr, const char *name) {
    auto tinfo = detail::get_type_info(typeid(T));
    return tinfo ? get_type_overload(this_ptr, tinfo, name) : function();
}

#define PYBIND11_OVERLOAD_INT(ret_type, cname, name, ...) { \
        pybind11::gil_scoped_acquire gil; \
        pybind11::function overload = pybind11::get_overload(static_cast<const cname *>(this), name); \
        if (overload) { \
            auto o = overload(__VA_ARGS__); \
            if (pybind11::detail::cast_is_temporary_value_reference<ret_type>::value) { \
                static pybind11::detail::overload_caster_t<ret_type> caster; \
                return pybind11::detail::cast_ref<ret_type>(std::move(o), caster); \
            } \
            else return pybind11::detail::cast_safe<ret_type>(std::move(o)); \
        } \
    }

/** \rst
    Macro to populate the virtual method in the trampoline class. This macro tries to look up a method named 'fn'
    from the Python side, deals with the :ref:`gil` and necessary argument conversions to call this method and return
    the appropriate type. See :ref:`overriding_virtuals` for more information. This macro should be used when the method
    name in C is not the same as the method name in Python. For example with `__str__`.

    .. code-block:: cpp

      std::string toString() override {
        PYBIND11_OVERLOAD_NAME(
            std::string, // Return type (ret_type)
            Animal,      // Parent class (cname)
            "__str__",   // Name of method in Python (name)
            toString,    // Name of function in C++ (fn)
        );
      }
\endrst */
#define PYBIND11_OVERLOAD_NAME(ret_type, cname, name, fn, ...) \
    PYBIND11_OVERLOAD_INT(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), name, __VA_ARGS__) \
    return cname::fn(__VA_ARGS__)

/** \rst
    Macro for pure virtual functions, this function is identical to :c:macro:`PYBIND11_OVERLOAD_NAME`, except that it
    throws if no overload can be found.
\endrst */
#define PYBIND11_OVERLOAD_PURE_NAME(ret_type, cname, name, fn, ...) \
    PYBIND11_OVERLOAD_INT(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), name, __VA_ARGS__) \
    pybind11::pybind11_fail("Tried to call pure virtual function \"" PYBIND11_STRINGIFY(cname) "::" name "\"");

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
              PYBIND11_OVERLOAD_PURE(
                  std::string, // Return type (ret_type)
                  Animal,      // Parent class (cname)
                  go,          // Name of function in C++ (must match Python name) (fn)
                  n_times      // Argument(s) (...)
              );
          }
      };
\endrst */
#define PYBIND11_OVERLOAD(ret_type, cname, fn, ...) \
    PYBIND11_OVERLOAD_NAME(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), #fn, fn, __VA_ARGS__)

/** \rst
    Macro for pure virtual functions, this function is identical to :c:macro:`PYBIND11_OVERLOAD`, except that it throws
    if no overload can be found.
\endrst */
#define PYBIND11_OVERLOAD_PURE(ret_type, cname, fn, ...) \
    PYBIND11_OVERLOAD_PURE_NAME(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), #fn, fn, __VA_ARGS__)

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

#if !PYBIND11_DECLARATIONS_ONLY
#include "pybind11.cpp"
#endif

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#  pragma warning(pop)
#elif defined(__GNUG__) && !defined(__clang__)
#  pragma GCC diagnostic pop
#endif
