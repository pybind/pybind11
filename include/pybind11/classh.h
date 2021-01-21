#pragma once

#include "detail/classh_type_casters.h"
#include "pybind11.h"
#include "smart_holder_poc.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

// clang-format off
template <typename type_, typename... options>
class classh : public detail::generic_type {
    template <typename T> using is_subtype = detail::is_strict_base_of<type_, T>;
    template <typename T> using is_base = detail::is_strict_base_of<T, type_>;
    // struct instead of using here to help MSVC:
    template <typename T> struct is_valid_class_option :
        detail::any_of<is_subtype<T>, is_base<T>> {};

public:
    using type = type_;
    using type_alias = detail::exactly_one_t<is_subtype, void, options...>;
    constexpr static bool has_alias = !std::is_void<type_alias>::value;
    using holder_type = pybindit::memory::smart_holder;

    static_assert(detail::all_of<is_valid_class_option<options>...>::value,
            "Unknown/invalid classh template parameters provided");

    static_assert(!has_alias || std::is_polymorphic<type>::value,
            "Cannot use an alias class with a non-polymorphic type");

    PYBIND11_OBJECT(classh, generic_type, PyType_Check)

    template <typename... Extra>
    classh(handle scope, const char *name, const Extra &... extra) {
        using namespace detail;

        // MI can only be specified via classh template options, not constructor parameters
        static_assert(
            none_of<is_pyobject<Extra>...>::value || // no base class arguments, or:
            (   constexpr_sum(is_pyobject<Extra>::value...) == 1 && // Exactly one base
                constexpr_sum(is_base<options>::value...)   == 0 && // no template option bases
                none_of<std::is_same<multiple_inheritance, Extra>...>::value), // no multiple_inheritance attr
            "Error: multiple inheritance bases must be specified via classh template options");

        type_record record;
        record.scope = scope;
        record.name = name;
        record.type = &typeid(type);
        record.type_size = sizeof(conditional_t<has_alias, type_alias, type>);
        record.type_align = alignof(conditional_t<has_alias, type_alias, type>&);
        record.holder_size = sizeof(holder_type);
        record.init_instance = init_instance;
        record.dealloc = dealloc;
        record.default_holder = false;

        set_operator_new<type>(&record);

        /* Register base classes specified via template arguments to classh, if any */
        PYBIND11_EXPAND_SIDE_EFFECTS(add_base<options>(record));

        /* Process optional arguments, if any */
        process_attributes<Extra...>::init(extra..., &record);

        generic_type::initialize(record, &modified_type_caster_generic_load_impl::local_load);

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
    classh &def(const char *name_, Func&& f, const Extra&... extra) {
        cpp_function cf(method_adaptor<type>(std::forward<Func>(f)), name(name_), is_method(*this),
                        sibling(getattr(*this, name_, none())), extra...);
        add_class_method(*this, name_, cf);
        return *this;
    }

    template <typename Func, typename... Extra> classh &
    def_static(const char *name_, Func &&f, const Extra&... extra) {
        static_assert(!std::is_member_function_pointer<Func>::value,
                "def_static(...) called with a non-static member function pointer");
        cpp_function cf(std::forward<Func>(f), name(name_), scope(*this),
                        sibling(getattr(*this, name_, none())), extra...);
        attr(cf.name()) = staticmethod(cf);
        return *this;
    }

    template <detail::op_id id, detail::op_type ot, typename L, typename R, typename... Extra>
    classh &def(const detail::op_<id, ot, L, R> &op, const Extra&... extra) {
        op.execute(*this, extra...);
        return *this;
    }

    template <detail::op_id id, detail::op_type ot, typename L, typename R, typename... Extra>
    classh & def_cast(const detail::op_<id, ot, L, R> &op, const Extra&... extra) {
        op.execute_cast(*this, extra...);
        return *this;
    }

    template <typename... Args, typename... Extra>
    classh &def(const detail::initimpl::constructor<Args...> &init, const Extra&... extra) {
        init.execute(*this, extra...);
        return *this;
    }

    template <typename... Args, typename... Extra>
    classh &def(const detail::initimpl::alias_constructor<Args...> &init, const Extra&... extra) {
        init.execute(*this, extra...);
        return *this;
    }

    template <typename... Args, typename... Extra>
    classh &def(detail::initimpl::factory<Args...> &&init, const Extra&... extra) {
        std::move(init).execute(*this, extra...);
        return *this;
    }

    template <typename... Args, typename... Extra>
    classh &def(detail::initimpl::pickle_factory<Args...> &&pf, const Extra &...extra) {
        std::move(pf).execute(*this, extra...);
        return *this;
    }

    template <typename Func>
    classh& def_buffer(Func &&func) {
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
    classh &def_buffer(Return (Class::*func)(Args...)) {
        return def_buffer([func] (type &obj) { return (obj.*func)(); });
    }

    template <typename Return, typename Class, typename... Args>
    classh &def_buffer(Return (Class::*func)(Args...) const) {
        return def_buffer([func] (const type &obj) { return (obj.*func)(); });
    }

    template <typename C, typename D, typename... Extra>
    classh &def_readwrite(const char *name, D C::*pm, const Extra&... extra) {
        static_assert(std::is_same<C, type>::value || std::is_base_of<C, type>::value, "def_readwrite() requires a class member (or base class member)");
        cpp_function fget([pm](const type &c) -> const D &{ return c.*pm; }, is_method(*this)),
                     fset([pm](type &c, const D &value) { c.*pm = value; }, is_method(*this));
        def_property(name, fget, fset, return_value_policy::reference_internal, extra...);
        return *this;
    }

    template <typename C, typename D, typename... Extra>
    classh &def_readonly(const char *name, const D C::*pm, const Extra& ...extra) {
        static_assert(std::is_same<C, type>::value || std::is_base_of<C, type>::value, "def_readonly() requires a class member (or base class member)");
        cpp_function fget([pm](const type &c) -> const D &{ return c.*pm; }, is_method(*this));
        def_property_readonly(name, fget, return_value_policy::reference_internal, extra...);
        return *this;
    }

    template <typename D, typename... Extra>
    classh &def_readwrite_static(const char *name, D *pm, const Extra& ...extra) {
        cpp_function fget([pm](object) -> const D &{ return *pm; }, scope(*this)),
                     fset([pm](object, const D &value) { *pm = value; }, scope(*this));
        def_property_static(name, fget, fset, return_value_policy::reference, extra...);
        return *this;
    }

    template <typename D, typename... Extra>
    classh &def_readonly_static(const char *name, const D *pm, const Extra& ...extra) {
        cpp_function fget([pm](object) -> const D &{ return *pm; }, scope(*this));
        def_property_readonly_static(name, fget, return_value_policy::reference, extra...);
        return *this;
    }

    /// Uses return_value_policy::reference_internal by default
    template <typename Getter, typename... Extra>
    classh &def_property_readonly(const char *name, const Getter &fget, const Extra& ...extra) {
        return def_property_readonly(name, cpp_function(method_adaptor<type>(fget)),
                                     return_value_policy::reference_internal, extra...);
    }

    /// Uses cpp_function's return_value_policy by default
    template <typename... Extra>
    classh &def_property_readonly(const char *name, const cpp_function &fget, const Extra& ...extra) {
        return def_property(name, fget, nullptr, extra...);
    }

    /// Uses return_value_policy::reference by default
    template <typename Getter, typename... Extra>
    classh &def_property_readonly_static(const char *name, const Getter &fget, const Extra& ...extra) {
        return def_property_readonly_static(name, cpp_function(fget), return_value_policy::reference, extra...);
    }

    /// Uses cpp_function's return_value_policy by default
    template <typename... Extra>
    classh &def_property_readonly_static(const char *name, const cpp_function &fget, const Extra& ...extra) {
        return def_property_static(name, fget, nullptr, extra...);
    }

    /// Uses return_value_policy::reference_internal by default
    template <typename Getter, typename Setter, typename... Extra>
    classh &def_property(const char *name, const Getter &fget, const Setter &fset, const Extra& ...extra) {
        return def_property(name, fget, cpp_function(method_adaptor<type>(fset)), extra...);
    }
    template <typename Getter, typename... Extra>
    classh &def_property(const char *name, const Getter &fget, const cpp_function &fset, const Extra& ...extra) {
        return def_property(name, cpp_function(method_adaptor<type>(fget)), fset,
                            return_value_policy::reference_internal, extra...);
    }

    /// Uses cpp_function's return_value_policy by default
    template <typename... Extra>
    classh &def_property(const char *name, const cpp_function &fget, const cpp_function &fset, const Extra& ...extra) {
        return def_property_static(name, fget, fset, is_method(*this), extra...);
    }

    /// Uses return_value_policy::reference by default
    template <typename Getter, typename... Extra>
    classh &def_property_static(const char *name, const Getter &fget, const cpp_function &fset, const Extra& ...extra) {
        return def_property_static(name, cpp_function(fget), fset, return_value_policy::reference, extra...);
    }

    /// Uses cpp_function's return_value_policy by default
    template <typename... Extra>
    classh &def_property_static(const char *name, const cpp_function &fget, const cpp_function &fset, const Extra& ...extra) {
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
    // clang-format on
    static void init_instance(detail::instance *inst, const void *holder_const_void_ptr) {
        // Need for const_cast is a consequence of the type_info::init_instance type:
        // void (*init_instance)(instance *, const void *);
        auto holder_void_ptr = const_cast<void *>(holder_const_void_ptr);

        auto v_h = inst->get_value_and_holder(detail::get_type_info(typeid(type)));
        if (!v_h.instance_registered()) {
            register_instance(inst, v_h.value_ptr(), v_h.type);
            v_h.set_instance_registered();
        }
        if (holder_void_ptr) {
            // Note: inst->owned ignored.
            auto holder_ptr = static_cast<holder_type *>(holder_void_ptr);
            new (std::addressof(v_h.holder<holder_type>())) holder_type(std::move(*holder_ptr));
        } else if (inst->owned) {
            new (std::addressof(v_h.holder<holder_type>()))
                holder_type(holder_type::from_raw_ptr_take_ownership(v_h.value_ptr<type>()));
        } else {
            new (std::addressof(v_h.holder<holder_type>()))
                holder_type(holder_type::from_raw_ptr_unowned(v_h.value_ptr<type>()));
        }
        v_h.set_holder_constructed();
    }
    // clang-format off

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

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
