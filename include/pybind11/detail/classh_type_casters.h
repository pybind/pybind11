#pragma once

#include "../cast.h"
#include "../pytypes.h"
#include "../smart_holder_poc.h"
#include "class.h"
#include "common.h"
#include "descr.h"
#include "internals.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace pybind11 {
namespace detail {

inline std::pair<bool, handle> find_existing_python_instance(void *src_void_ptr,
                                                             const detail::type_info *tinfo) {
    // Loop copied from type_caster_generic::cast.
    // IMPROVEABLE: Factor out of type_caster_generic::cast.
    auto it_instances = get_internals().registered_instances.equal_range(src_void_ptr);
    for (auto it_i = it_instances.first; it_i != it_instances.second; ++it_i) {
        for (auto instance_type : detail::all_type_info(Py_TYPE(it_i->second))) {
            if (instance_type && same_type(*instance_type->cpptype, *tinfo->cpptype))
                return std::make_pair(true, handle((PyObject *) it_i->second).inc_ref());
        }
    }
    return std::make_pair(false, handle());
}

// clang-format off
class modified_type_caster_generic_load_impl {
public:
    PYBIND11_NOINLINE modified_type_caster_generic_load_impl(const std::type_info &type_info)
        : typeinfo(get_type_info(type_info)), cpptype(&type_info) { }

    explicit modified_type_caster_generic_load_impl(const type_info *typeinfo = nullptr)
        : typeinfo(typeinfo), cpptype(typeinfo ? typeinfo->cpptype : nullptr) { }

    bool load(handle src, bool convert) {
        return load_impl<modified_type_caster_generic_load_impl>(src, convert);
    }

    // Base methods for generic caster; there are overridden in copyable_holder_caster
    void load_value_and_holder(value_and_holder &&v_h) {
        loaded_v_h = std::move(v_h);
        if (!loaded_v_h.holder_constructed()) {
            // IMPROVEABLE: Error message. A change to the existing internals is
            // needed to cleanly distinguish between uninitialized or disowned.
            throw std::runtime_error("Missing value for wrapped C++ type:"
                                     " Python instance is uninitialized or was disowned.");
        }
        if (v_h.value_ptr() == nullptr) {
            pybind11_fail("classh_type_casters: Unexpected v_h.value_ptr() nullptr.");
        }
        loaded_v_h.type = typeinfo;
    }

    bool try_implicit_casts(handle src, bool convert) {
        for (auto &cast : typeinfo->implicit_casts) {
            modified_type_caster_generic_load_impl sub_caster(*cast.first);
            if (sub_caster.load(src, convert)) {
                if (loaded_v_h_cpptype != nullptr) {
                    pybind11_fail("classh_type_casters: try_implicit_casts failure.");
                }
                loaded_v_h = sub_caster.loaded_v_h;
                loaded_v_h_cpptype = cast.first;
                implicit_cast = cast.second;
                return true;
            }
        }
        return false;
    }

    bool try_direct_conversions(handle src) {
        for (auto &converter : *typeinfo->direct_conversions) {
            if (converter(src.ptr(), loaded_v_h.value_ptr()))
                return true;
        }
        return false;
    }

    PYBIND11_NOINLINE static void *local_load(PyObject *src, const type_info *ti) {
        // Not thread safe. But the GIL needs to be held anyway in the context of this code.
        static modified_type_caster_generic_load_impl caster;
        caster = modified_type_caster_generic_load_impl(ti);
        if (caster.load(src, false)) {
            // Trick to work with the existing pybind11 internals.
            return &caster; // Any pointer except nullptr;
        }
        return nullptr;
    }

    /// Try to load with foreign typeinfo, if available. Used when there is no
    /// native typeinfo, or when the native one wasn't able to produce a value.
    PYBIND11_NOINLINE bool try_load_foreign_module_local(handle src) {
        constexpr auto *local_key = PYBIND11_MODULE_LOCAL_ID;
        const auto pytype = type::handle_of(src);
        if (!hasattr(pytype, local_key))
            return false;

        type_info *foreign_typeinfo = reinterpret_borrow<capsule>(getattr(pytype, local_key));
        // Only consider this foreign loader if actually foreign and is a loader of the correct cpp type
        if (foreign_typeinfo->module_local_load == &local_load
            || (cpptype && !same_type(*cpptype, *foreign_typeinfo->cpptype)))
            return false;

        void* void_ptr = foreign_typeinfo->module_local_load(src.ptr(), foreign_typeinfo);
        if (void_ptr != nullptr) {
            auto foreign_load_impl = static_cast<modified_type_caster_generic_load_impl *>(void_ptr);
            if (loaded_v_h_cpptype != nullptr) {
                pybind11_fail("classh_type_casters: try_load_foreign_module_local failure.");
            }
            loaded_v_h = foreign_load_impl->loaded_v_h;
            loaded_v_h_cpptype = foreign_load_impl->loaded_v_h_cpptype;
            implicit_cast = foreign_load_impl->implicit_cast;
            return true;
        }
        return false;
    }

    // Implementation of `load`; this takes the type of `this` so that it can dispatch the relevant
    // bits of code between here and copyable_holder_caster where the two classes need different
    // logic (without having to resort to virtual inheritance).
    template <typename ThisT>
    PYBIND11_NOINLINE bool load_impl(handle src, bool convert) {
        if (!src) return false;
        if (!typeinfo) return try_load_foreign_module_local(src);
        if (src.is_none()) {
            // Defer accepting None to other overloads (if we aren't in convert mode):
            if (!convert) return false;
            loaded_v_h = value_and_holder();
            return true;
        }

        auto &this_ = static_cast<ThisT &>(*this);

        PyTypeObject *srctype = Py_TYPE(src.ptr());

        // Case 1: If src is an exact type match for the target type then we can reinterpret_cast
        // the instance's value pointer to the target type:
        if (srctype == typeinfo->type) {
            this_.load_value_and_holder(reinterpret_cast<instance *>(src.ptr())->get_value_and_holder());
            return true;
        }
        // Case 2: We have a derived class
        else if (PyType_IsSubtype(srctype, typeinfo->type)) {
            auto &bases = all_type_info(srctype); // subtype bases
            bool no_cpp_mi = typeinfo->simple_type;

            // Case 2a: the python type is a Python-inherited derived class that inherits from just
            // one simple (no MI) pybind11 class, or is an exact match, so the C++ instance is of
            // the right type and we can use reinterpret_cast.
            // (This is essentially the same as case 2b, but because not using multiple inheritance
            // is extremely common, we handle it specially to avoid the loop iterator and type
            // pointer lookup overhead)
            if (bases.size() == 1 && (no_cpp_mi || bases.front()->type == typeinfo->type)) {
                this_.load_value_and_holder(reinterpret_cast<instance *>(src.ptr())->get_value_and_holder());
                loaded_v_h_cpptype = bases.front()->cpptype;
                reinterpret_cast_deemed_ok = true;
                return true;
            }
            // Case 2b: the python type inherits from multiple C++ bases.  Check the bases to see if
            // we can find an exact match (or, for a simple C++ type, an inherited match); if so, we
            // can safely reinterpret_cast to the relevant pointer.
            else if (bases.size() > 1) {
                for (auto base : bases) {
                    if (no_cpp_mi ? PyType_IsSubtype(base->type, typeinfo->type) : base->type == typeinfo->type) {
                        this_.load_value_and_holder(reinterpret_cast<instance *>(src.ptr())->get_value_and_holder(base));
                        loaded_v_h_cpptype = base->cpptype;
                        reinterpret_cast_deemed_ok = true;
                        return true;
                    }
                }
            }

            // Case 2c: C++ multiple inheritance is involved and we couldn't find an exact type match
            // in the registered bases, above, so try implicit casting (needed for proper C++ casting
            // when MI is involved).
            if (this_.try_implicit_casts(src, convert)) {
                return true;
            }
        }

        // Perform an implicit conversion
        if (convert) {
            for (auto &converter : typeinfo->implicit_conversions) {
                auto temp = reinterpret_steal<object>(converter(src.ptr(), typeinfo->type));
                if (load_impl<ThisT>(temp, false)) {
                    loader_life_support::add_patient(temp);
                    return true;
                }
            }
            if (this_.try_direct_conversions(src))
                return true;
        }

        // Failed to match local typeinfo. Try again with global.
        if (typeinfo->module_local) {
            if (auto gtype = get_global_type_info(*typeinfo->cpptype)) {
                typeinfo = gtype;
                return load(src, false);
            }
        }

        // Global typeinfo has precedence over foreign module_local
        return try_load_foreign_module_local(src);
    }

    const type_info *typeinfo = nullptr;
    const std::type_info *cpptype = nullptr;
    const std::type_info *loaded_v_h_cpptype = nullptr;
    void *(*implicit_cast)(void *) = nullptr;
    value_and_holder loaded_v_h;
    bool reinterpret_cast_deemed_ok = false;
};
// clang-format on

template <typename T>
struct smart_holder_type_caster_load {
    using holder_type = pybindit::memory::smart_holder;

    bool load(handle src, bool convert) {
        load_impl = modified_type_caster_generic_load_impl(typeid(T));
        if (!load_impl.load(src, convert))
            return false;
        loaded_smhldr_ptr = &load_impl.loaded_v_h.holder<holder_type>();
        return true;
    }

    T *as_raw_ptr_unowned() {
        if (load_impl.loaded_v_h_cpptype != nullptr) {
            if (load_impl.reinterpret_cast_deemed_ok) {
                return static_cast<T *>(loaded_smhldr_ptr->vptr.get());
            }
            if (load_impl.implicit_cast != nullptr) {
                void *implicit_casted = load_impl.implicit_cast(loaded_smhldr_ptr->vptr.get());
                return static_cast<T *>(implicit_casted);
            }
        }
        return loaded_smhldr_ptr->as_raw_ptr_unowned<T>();
    }

    std::unique_ptr<T> loaded_as_unique_ptr() {
        void *value_void_ptr = load_impl.loaded_v_h.value_ptr();
        auto unq_ptr         = loaded_smhldr_ptr->as_unique_ptr<T>();
        load_impl.loaded_v_h.holder<holder_type>().~holder_type();
        load_impl.loaded_v_h.set_holder_constructed(false);
        load_impl.loaded_v_h.value_ptr() = nullptr;
        deregister_instance(load_impl.loaded_v_h.inst, value_void_ptr, load_impl.loaded_v_h.type);
        return unq_ptr;
    }

protected:
    modified_type_caster_generic_load_impl load_impl;
    holder_type *loaded_smhldr_ptr = nullptr;
};

// type_caster_base BEGIN
// clang-format off
// Helper factored out of type_caster_base.
struct make_constructor {
    using Constructor = void *(*)(const void *);

    /* Only enabled when the types are {copy,move}-constructible *and* when the type
       does not have a private operator new implementation. */
    template <typename T, typename = enable_if_t<is_copy_constructible<T>::value>>
    static auto make_copy_constructor(const T *x) -> decltype(new T(*x), Constructor{}) {
        return [](const void *arg) -> void * {
            return new T(*reinterpret_cast<const T *>(arg));
        };
    }

    template <typename T, typename = enable_if_t<std::is_move_constructible<T>::value>>
    static auto make_move_constructor(const T *x) -> decltype(new T(std::move(*const_cast<T *>(x))), Constructor{}) {
        return [](const void *arg) -> void * {
            return new T(std::move(*const_cast<T *>(reinterpret_cast<const T *>(arg))));
        };
    }

    static Constructor make_copy_constructor(...) { return nullptr; }
    static Constructor make_move_constructor(...) { return nullptr; }
};
// clang-format on
// type_caster_base END

template <typename T>
struct classh_type_caster : smart_holder_type_caster_load<T> {
    static constexpr auto name = _<T>();

    // static handle cast(T, ...)
    // is redundant (leads to ambiguous overloads).

    static handle cast(T &&src, return_value_policy /*policy*/, handle parent) {
        // type_caster_base BEGIN
        // clang-format off
        return cast(&src, return_value_policy::move, parent);
        // clang-format on
        // type_caster_base END
    }

    static handle cast(T const &src, return_value_policy policy, handle parent) {
        // type_caster_base BEGIN
        // clang-format off
        if (policy == return_value_policy::automatic || policy == return_value_policy::automatic_reference)
            policy = return_value_policy::copy;
        return cast(&src, policy, parent);
        // clang-format on
        // type_caster_base END
    }

    static handle cast(T &src, return_value_policy policy, handle parent) {
        return cast(const_cast<T const &>(src), policy, parent); // Mutbl2Const
    }

    static handle cast(T const *src, return_value_policy policy, handle parent) {
        auto st = type_caster_base<T>::src_and_type(src);
        return cast_const_raw_ptr( // Originally type_caster_generic::cast.
            st.first,
            policy,
            parent,
            st.second,
            make_constructor::make_copy_constructor(src),
            make_constructor::make_move_constructor(src));
    }

    static handle cast(T *src, return_value_policy policy, handle parent) {
        return cast(const_cast<T const *>(src), policy, parent); // Mutbl2Const
    }

    template <typename T_>
    using cast_op_type = conditional_t<
        std::is_same<remove_reference_t<T_>, T const *>::value,
        T const *,
        conditional_t<
            std::is_same<remove_reference_t<T_>, T *>::value,
            T *,
            conditional_t<std::is_same<T_, T const &>::value,
                          T const &,
                          conditional_t<std::is_same<T_, T &>::value,
                                        T &,
                                        conditional_t<std::is_same<T_, T &&>::value, T &&, T>>>>>;

    // clang-format off

    operator T()        { return this->loaded_smhldr_ptr->template lvalue_ref<T>(); }
    operator T&&() &&   { return this->loaded_smhldr_ptr->template rvalue_ref<T>(); }
    operator T const&() { return this->loaded_smhldr_ptr->template lvalue_ref<T>(); }
    operator T&()       { return this->loaded_smhldr_ptr->template lvalue_ref<T>(); }
    operator T const*() { return this->as_raw_ptr_unowned(); }
    operator T*()       { return this->as_raw_ptr_unowned(); }

    // clang-format on

    // Originally type_caster_generic::cast.
    PYBIND11_NOINLINE static handle cast_const_raw_ptr(const void *_src,
                                                       return_value_policy policy,
                                                       handle parent,
                                                       const detail::type_info *tinfo,
                                                       void *(*copy_constructor)(const void *),
                                                       void *(*move_constructor)(const void *),
                                                       const void *existing_holder = nullptr) {
        if (!tinfo) // no type info: error will be set already
            return handle();

        void *src = const_cast<void *>(_src);
        if (src == nullptr)
            return none().release();

        auto existing_inst = find_existing_python_instance(src, tinfo);
        if (existing_inst.first)
            return existing_inst.second;

        auto inst       = reinterpret_steal<object>(make_new_instance(tinfo->type));
        auto wrapper    = reinterpret_cast<instance *>(inst.ptr());
        wrapper->owned  = false;
        void *&valueptr = values_and_holders(wrapper).begin()->value_ptr();

        switch (policy) {
            case return_value_policy::automatic:
            case return_value_policy::take_ownership:
                valueptr       = src;
                wrapper->owned = true;
                break;

            case return_value_policy::automatic_reference:
            case return_value_policy::reference:
                valueptr       = src;
                wrapper->owned = false;
                break;

            case return_value_policy::copy:
                if (copy_constructor)
                    valueptr = copy_constructor(src);
                else {
#if defined(NDEBUG)
                    throw cast_error("return_value_policy = copy, but type is "
                                     "non-copyable! (compile in debug mode for details)");
#else
                    std::string type_name(tinfo->cpptype->name());
                    detail::clean_type_id(type_name);
                    throw cast_error("return_value_policy = copy, but type " + type_name
                                     + " is non-copyable!");
#endif
                }
                wrapper->owned = true;
                break;

            case return_value_policy::move:
                if (move_constructor)
                    valueptr = move_constructor(src);
                else if (copy_constructor)
                    valueptr = copy_constructor(src);
                else {
#if defined(NDEBUG)
                    throw cast_error("return_value_policy = move, but type is neither "
                                     "movable nor copyable! "
                                     "(compile in debug mode for details)");
#else
                    std::string type_name(tinfo->cpptype->name());
                    detail::clean_type_id(type_name);
                    throw cast_error("return_value_policy = move, but type " + type_name
                                     + " is neither movable nor copyable!");
#endif
                }
                wrapper->owned = true;
                break;

            case return_value_policy::reference_internal:
                valueptr       = src;
                wrapper->owned = false;
                keep_alive_impl(inst, parent);
                break;

            default:
                throw cast_error("unhandled return_value_policy: should not happen!");
        }

        tinfo->init_instance(wrapper, existing_holder);

        return inst.release();
    }
};

template <typename T>
struct classh_type_caster<std::shared_ptr<T>> : smart_holder_type_caster_load<T> {
    static constexpr auto name = _<std::shared_ptr<T>>();

    static handle cast(const std::shared_ptr<T> &src, return_value_policy policy, handle parent) {
        if (policy != return_value_policy::automatic
            && policy != return_value_policy::reference_internal) {
            // IMPROVEABLE: Error message.
            throw cast_error("Invalid return_value_policy for shared_ptr.");
        }

        auto src_raw_ptr = src.get();
        auto st          = type_caster_base<T>::src_and_type(src_raw_ptr);
        if (st.first == nullptr)
            return none().release(); // PyErr was set already.

        void *src_raw_void_ptr         = static_cast<void *>(src_raw_ptr);
        const detail::type_info *tinfo = st.second;
        auto existing_inst             = find_existing_python_instance(src_raw_void_ptr, tinfo);
        if (existing_inst.first)
            // MISSING: Enforcement of consistency with existing smart_holder.
            // MISSING: keep_alive.
            return existing_inst.second;

        object inst            = reinterpret_steal<object>(make_new_instance(tinfo->type));
        instance *inst_raw_ptr = reinterpret_cast<instance *>(inst.ptr());
        inst_raw_ptr->owned    = true;
        void *&valueptr        = values_and_holders(inst_raw_ptr).begin()->value_ptr();
        valueptr               = src_raw_void_ptr;

        auto smhldr = pybindit::memory::smart_holder::from_shared_ptr(src);
        tinfo->init_instance(inst_raw_ptr, static_cast<const void *>(&smhldr));

        if (policy == return_value_policy::reference_internal)
            keep_alive_impl(inst, parent);

        return inst.release();
    }

    template <typename>
    using cast_op_type = std::shared_ptr<T>;

    operator std::shared_ptr<T>() { return this->loaded_smhldr_ptr->template as_shared_ptr<T>(); }
};

template <typename T>
struct classh_type_caster<std::shared_ptr<T const>> : smart_holder_type_caster_load<T> {
    static constexpr auto name = _<std::shared_ptr<T const>>();

    static handle
    cast(const std::shared_ptr<T const> &src, return_value_policy policy, handle parent) {
        return type_caster<std::shared_ptr<T>>::cast(
            std::const_pointer_cast<T>(src), // Const2Mutbl
            policy,
            parent);
    }

    template <typename>
    using cast_op_type = std::shared_ptr<T const>;

    operator std::shared_ptr<T const>() {
        return this->loaded_smhldr_ptr->template as_shared_ptr<T>();
    }
};

template <typename T>
struct classh_type_caster<std::unique_ptr<T>> : smart_holder_type_caster_load<T> {
    static constexpr auto name = _<std::unique_ptr<T>>();

    static handle cast(std::unique_ptr<T> &&src, return_value_policy policy, handle parent) {
        if (policy != return_value_policy::automatic
            && policy != return_value_policy::reference_internal) {
            // IMPROVEABLE: Error message.
            throw cast_error("Invalid return_value_policy for unique_ptr.");
        }

        auto src_raw_ptr = src.get();
        auto st          = type_caster_base<T>::src_and_type(src_raw_ptr);
        if (st.first == nullptr)
            return none().release(); // PyErr was set already.

        void *src_raw_void_ptr         = static_cast<void *>(src_raw_ptr);
        const detail::type_info *tinfo = st.second;
        auto existing_inst             = find_existing_python_instance(src_raw_void_ptr, tinfo);
        if (existing_inst.first)
            throw cast_error("Invalid unique_ptr: another instance owns this pointer already.");

        object inst            = reinterpret_steal<object>(make_new_instance(tinfo->type));
        instance *inst_raw_ptr = reinterpret_cast<instance *>(inst.ptr());
        inst_raw_ptr->owned    = true;
        void *&valueptr        = values_and_holders(inst_raw_ptr).begin()->value_ptr();
        valueptr               = src_raw_void_ptr;

        auto smhldr = pybindit::memory::smart_holder::from_unique_ptr(std::move(src));
        tinfo->init_instance(inst_raw_ptr, static_cast<const void *>(&smhldr));

        if (policy == return_value_policy::reference_internal)
            keep_alive_impl(inst, parent);

        return inst.release();
    }

    template <typename>
    using cast_op_type = std::unique_ptr<T>;

    operator std::unique_ptr<T>() { return this->loaded_as_unique_ptr(); }
};

template <typename T>
struct classh_type_caster<std::unique_ptr<T const>> : smart_holder_type_caster_load<T> {
    static constexpr auto name = _<std::unique_ptr<T const>>();

    static handle cast(std::unique_ptr<T const> &&src, return_value_policy policy, handle parent) {
        return type_caster<std::unique_ptr<T>>::cast(
            std::unique_ptr<T>(const_cast<T *>(src.release())), // Const2Mutbl
            policy,
            parent);
    }

    template <typename>
    using cast_op_type = std::unique_ptr<T const>;

    operator std::unique_ptr<T const>() { return this->loaded_as_unique_ptr(); }
};

#define PYBIND11_CLASSH_TYPE_CASTERS(T)                                                           \
    namespace pybind11 {                                                                          \
    namespace detail {                                                                            \
    template <>                                                                                   \
    class type_caster<T> : public classh_type_caster<T> {};                                       \
    template <>                                                                                   \
    class type_caster<std::shared_ptr<T>> : public classh_type_caster<std::shared_ptr<T>> {};     \
    template <>                                                                                   \
    class type_caster<std::shared_ptr<T const>>                                                   \
        : public classh_type_caster<std::shared_ptr<T const>> {};                                 \
    template <>                                                                                   \
    class type_caster<std::unique_ptr<T>> : public classh_type_caster<std::unique_ptr<T>> {};     \
    template <>                                                                                   \
    class type_caster<std::unique_ptr<T const>>                                                   \
        : public classh_type_caster<std::unique_ptr<T const>> {};                                 \
    }                                                                                             \
    }

} // namespace detail
} // namespace pybind11
