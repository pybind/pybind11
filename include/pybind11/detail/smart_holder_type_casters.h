// Copyright (c) 2021 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include "../gil.h"
#include "../pytypes.h"
#include "common.h"
#include "descr.h"
#include "internals.h"
#include "smart_holder_poc.h"
#include "smart_holder_sfinae_hooks_only.h"
#include "type_caster_base.h"
#include "typeid.h"
#include "virtual_overrider_self_life_support.h"

#include <cstddef>
#include <memory>
#include <new>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

using pybindit::memory::smart_holder;

PYBIND11_NAMESPACE_BEGIN(detail)

template <>
struct is_smart_holder_type<smart_holder> : std::true_type {};

// SMART_HOLDER_WIP: Needs refactoring of existing pybind11 code.
inline void register_instance(instance *self, void *valptr, const type_info *tinfo);
inline bool deregister_instance(instance *self, void *valptr, const type_info *tinfo);

// The modified_type_caster_generic_load_impl could replace type_caster_generic::load_impl but not
// vice versa. The main difference is that the original code only propagates a reference to the
// held value, while the modified implementation propagates value_and_holder.
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
        if (!v_h.holder_constructed()) {
            // This is needed for old-style __init__.
            // type_caster_generic::load_value BEGIN
            auto *&vptr = v_h.value_ptr();
            // Lazy allocation for unallocated values:
            if (vptr == nullptr) {
                // Lazy allocation for unallocated values:
                auto *type = v_h.type ? v_h.type : typeinfo;
                if (type->operator_new) {
                    vptr = type->operator_new(type->type_size);
                } else {
                    #if defined(__cpp_aligned_new) && (!defined(_MSC_VER) || _MSC_VER >= 1912)
                        if (type->type_align > __STDCPP_DEFAULT_NEW_ALIGNMENT__)
                            vptr = ::operator new(type->type_size,
                                                  std::align_val_t(type->type_align));
                        else
                    #endif
                    vptr = ::operator new(type->type_size);
                }
            }
            // type_caster_generic::load_value END
        }
        loaded_v_h = std::move(v_h);
        loaded_v_h.type = typeinfo;
    }

    bool try_implicit_casts(handle src, bool convert) {
        for (auto &cast : typeinfo->implicit_casts) {
            modified_type_caster_generic_load_impl sub_caster(*cast.first);
            if (sub_caster.load(src, convert)) {
                if (loaded_v_h_cpptype != nullptr) {
                    pybind11_fail("smart_holder_type_casters: try_implicit_casts failure.");
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
            if (converter(src.ptr(), unowned_void_ptr_from_direct_conversion)) {
                return true;
            }
        }
        return false;
    }

    PYBIND11_NOINLINE static void *local_load(PyObject *src, const type_info *ti) {
        std::unique_ptr<modified_type_caster_generic_load_impl> loader(
            new modified_type_caster_generic_load_impl(ti));
        if (loader->load(src, false)) {
            // Trick to work with the existing pybind11 internals.
            // The void pointer is immediately captured in a new unique_ptr in
            // try_load_foreign_module_local. If this assumption is violated sanitizers
            // will most likely flag a leak (verified to be the case with ASAN).
            return static_cast<void *>(loader.release());
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

        void* foreign_loader_void_ptr =
            foreign_typeinfo->module_local_load(src.ptr(), foreign_typeinfo);
        if (foreign_loader_void_ptr != nullptr) {
            auto foreign_loader = std::unique_ptr<modified_type_caster_generic_load_impl>(
                static_cast<modified_type_caster_generic_load_impl *>(foreign_loader_void_ptr));
            // Magic number intentionally hard-coded for simplicity and maximum robustness.
            if (foreign_loader->local_load_safety_guard != 1887406645) {
                pybind11_fail(
                    "smart_holder_type_casters: Unexpected local_load_safety_guard,"
                    " possibly due to py::class_ holder mixup.");
            }
            if (loaded_v_h_cpptype != nullptr) {
                pybind11_fail("smart_holder_type_casters: try_load_foreign_module_local failure.");
            }
            loaded_v_h = foreign_loader->loaded_v_h;
            loaded_v_h_cpptype = foreign_loader->loaded_v_h_cpptype;
            implicit_cast = foreign_loader->implicit_cast;
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
    void *unowned_void_ptr_from_direct_conversion = nullptr;
    const std::type_info *loaded_v_h_cpptype = nullptr;
    void *(*implicit_cast)(void *) = nullptr;
    value_and_holder loaded_v_h;
    bool reinterpret_cast_deemed_ok = false;
    // Magic number intentionally hard-coded, to guard against class_ holder mixups.
    // Ideally type_caster_generic would have a similar guard, but this requires a change there.
    // SMART_HOLDER_WIP: If it is decided that this guard is useful long term, potentially
    // set/reset this value in ctor/dtor, mark volatile.
    std::size_t local_load_safety_guard = 1887406645; // 32-bit compatible value for portability.
};
// clang-format on

struct smart_holder_type_caster_class_hooks : smart_holder_type_caster_base_tag {
    static decltype(&modified_type_caster_generic_load_impl::local_load)
    get_local_load_function_ptr() {
        return &modified_type_caster_generic_load_impl::local_load;
    }

    template <typename T>
    static void init_instance_for_type(detail::instance *inst,
                                       const void *holder_const_void_ptr,
                                       bool has_alias) {
        // Need for const_cast is a consequence of the type_info::init_instance type:
        // void (*init_instance)(instance *, const void *);
        auto holder_void_ptr = const_cast<void *>(holder_const_void_ptr);

        auto v_h = inst->get_value_and_holder(detail::get_type_info(typeid(T)));
        if (!v_h.instance_registered()) {
            register_instance(inst, v_h.value_ptr(), v_h.type);
            v_h.set_instance_registered();
        }
        using holder_type = pybindit::memory::smart_holder;
        if (holder_void_ptr) {
            // Note: inst->owned ignored.
            auto holder_ptr = static_cast<holder_type *>(holder_void_ptr);
            new (std::addressof(v_h.holder<holder_type>())) holder_type(std::move(*holder_ptr));
        } else if (inst->owned) {
            new (std::addressof(v_h.holder<holder_type>()))
                holder_type(holder_type::from_raw_ptr_take_ownership(v_h.value_ptr<T>()));
        } else {
            new (std::addressof(v_h.holder<holder_type>()))
                holder_type(holder_type::from_raw_ptr_unowned(v_h.value_ptr<T>()));
        }
        v_h.holder<holder_type>().pointee_depends_on_holder_owner = has_alias;
        v_h.set_holder_constructed();
    }

    template <typename T, typename D>
    static smart_holder smart_holder_from_unique_ptr(std::unique_ptr<T, D> &&unq_ptr) {
        return pybindit::memory::smart_holder::from_unique_ptr(std::move(unq_ptr));
    }

    template <typename T>
    static smart_holder smart_holder_from_shared_ptr(std::shared_ptr<T> shd_ptr) {
        return pybindit::memory::smart_holder::from_shared_ptr(shd_ptr);
    }
};

template <typename T>
struct smart_holder_type_caster_load {
    using holder_type = pybindit::memory::smart_holder;

    bool load(handle src, bool convert) {
        static_assert(type_uses_smart_holder_type_caster<T>::value, "Internal consistency error.");
        load_impl = modified_type_caster_generic_load_impl(typeid(T));
        if (!load_impl.load(src, convert))
            return false;
        return true;
    }

    T *loaded_as_raw_ptr_unowned() const {
        void *void_ptr = load_impl.unowned_void_ptr_from_direct_conversion;
        if (void_ptr == nullptr) {
            if (have_holder()) {
                throw_if_uninitialized_or_disowned_holder();
                void_ptr = holder().template as_raw_ptr_unowned<void>();
            } else if (load_impl.loaded_v_h.vh != nullptr)
                void_ptr = load_impl.loaded_v_h.value_ptr();
            if (void_ptr == nullptr)
                return nullptr;
        }
        return convert_type(void_ptr);
    }

    T &loaded_as_lvalue_ref() const {
        T *raw_ptr = loaded_as_raw_ptr_unowned();
        if (raw_ptr == nullptr)
            throw reference_cast_error();
        return *raw_ptr;
    }

    struct shared_ptr_dec_ref_deleter {
        // Note: deleter destructor fails on MSVC 2015 and GCC 4.8, so we manually call dec_ref
        // here instead.
        handle ref;
        void operator()(void *) {
            gil_scoped_acquire gil;
            ref.dec_ref();
        }
    };

    std::shared_ptr<T> loaded_as_shared_ptr() const {
        if (load_impl.unowned_void_ptr_from_direct_conversion != nullptr)
            throw cast_error("Unowned pointer from direct conversion cannot be converted to a"
                             " std::shared_ptr.");
        if (!have_holder())
            return nullptr;
        throw_if_uninitialized_or_disowned_holder();
        holder().ensure_was_not_disowned("loaded_as_shared_ptr");
        auto void_raw_ptr = holder().template as_raw_ptr_unowned<void>();
        auto type_raw_ptr = convert_type(void_raw_ptr);
        if (holder().pointee_depends_on_holder_owner) {
            // Tie lifetime of trampoline Python part to C++ part (PR #2839).
            return std::shared_ptr<T>(
                type_raw_ptr,
                shared_ptr_dec_ref_deleter{
                    handle((PyObject *) load_impl.loaded_v_h.inst).inc_ref()});
        }
        std::shared_ptr<void> void_shd_ptr = holder().template as_shared_ptr<void>();
        return std::shared_ptr<T>(void_shd_ptr, type_raw_ptr);
    }

    template <typename D>
    std::unique_ptr<T, D> loaded_as_unique_ptr(const char *context = "loaded_as_unique_ptr") {
        if (load_impl.unowned_void_ptr_from_direct_conversion != nullptr)
            throw cast_error("Unowned pointer from direct conversion cannot be converted to a"
                             " std::unique_ptr.");
        if (!have_holder())
            return nullptr;
        throw_if_uninitialized_or_disowned_holder();
        holder().ensure_was_not_disowned(context);
        holder().template ensure_compatible_rtti_uqp_del<T, D>(context);
        holder().ensure_use_count_1(context);
        auto raw_void_ptr = holder().template as_raw_ptr_unowned<void>();

        void *value_void_ptr = load_impl.loaded_v_h.value_ptr();
        if (value_void_ptr != raw_void_ptr) {
            pybind11_fail("smart_holder_type_casters: loaded_as_unique_ptr failure:"
                          " value_void_ptr != raw_void_ptr");
        }

        // SMART_HOLDER_WIP: MISSING: Safety checks for type conversions
        // (T must be polymorphic or meet certain other conditions).
        T *raw_type_ptr = convert_type(raw_void_ptr);

        auto *self_life_support
            = dynamic_cast_virtual_overrider_self_life_support_ptr(raw_type_ptr);
        if (self_life_support == nullptr && holder().pointee_depends_on_holder_owner) {
            throw value_error("Ownership of instance with virtual overrides in Python cannot be "
                              "transferred to C++.");
        }

        // Critical transfer-of-ownership section. This must stay together.
        if (self_life_support != nullptr) {
            holder().disown();
        } else {
            holder().release_ownership();
        }
        auto result = std::unique_ptr<T, D>(raw_type_ptr);
        if (self_life_support != nullptr) {
            Py_INCREF((PyObject *) load_impl.loaded_v_h.inst);
            self_life_support->loaded_v_h = load_impl.loaded_v_h;
        } else {
            load_impl.loaded_v_h.value_ptr() = nullptr;
            deregister_instance(
                load_impl.loaded_v_h.inst, value_void_ptr, load_impl.loaded_v_h.type);
        }
        // Critical section end.

        return result;
    }

private:
    modified_type_caster_generic_load_impl load_impl;

    bool have_holder() const {
        return load_impl.loaded_v_h.vh != nullptr && load_impl.loaded_v_h.holder_constructed();
    }

    holder_type &holder() const { return load_impl.loaded_v_h.holder<holder_type>(); }

    // have_holder() must be true or this function will fail.
    void throw_if_uninitialized_or_disowned_holder() const {
        if (!holder().is_populated) {
            pybind11_fail("Missing value for wrapped C++ type:"
                          " Python instance is uninitialized.");
        }
        if (!holder().has_pointee()) {
            throw value_error("Missing value for wrapped C++ type:"
                              " Python instance was disowned.");
        }
    }

    T *convert_type(void *void_ptr) const {
        if (void_ptr != nullptr && load_impl.loaded_v_h_cpptype != nullptr
            && !load_impl.reinterpret_cast_deemed_ok && load_impl.implicit_cast != nullptr) {
            void_ptr = load_impl.implicit_cast(void_ptr);
        }
        return static_cast<T *>(void_ptr);
    }
};

// SMART_HOLDER_WIP: Needs refactoring of existing pybind11 code.
struct make_constructor : private type_caster_base<int> { // Any type, nothing special about int.
    using type_caster_base<int>::Constructor;
    using type_caster_base<int>::make_copy_constructor;
    using type_caster_base<int>::make_move_constructor;
};

template <typename T>
struct smart_holder_type_caster : smart_holder_type_caster_load<T>,
                                  smart_holder_type_caster_class_hooks {
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

#if defined(_MSC_VER) && _MSC_VER < 1910
    // Working around MSVC 2015 bug. const-correctness is lost.
    // SMART_HOLDER_WIP: IMPROVABLE: make common code work with MSVC 2015.
    template <typename T_>
    using cast_op_type = detail::cast_op_type<T_>;
#else
    template <typename T_>
    using cast_op_type = conditional_t<
        std::is_same<remove_reference_t<T_>, T const *>::value,
        T const *,
        conditional_t<std::is_same<remove_reference_t<T_>, T *>::value,
                      T *,
                      conditional_t<std::is_same<T_, T const &>::value, T const &, T &>>>;
#endif

    // The const operators here prove that the existing type_caster mechanism already supports
    // const-correctness. However, fully implementing const-correctness inside this type_caster
    // is still a major project.
    operator T const &() const {
        return const_cast<smart_holder_type_caster *>(this)->loaded_as_lvalue_ref();
    }
    operator T const *() const {
        return const_cast<smart_holder_type_caster *>(this)->loaded_as_raw_ptr_unowned();
    }
    operator T &() { return this->loaded_as_lvalue_ref(); }
    operator T *() { return this->loaded_as_raw_ptr_unowned(); }

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

        if (handle existing_inst = find_registered_python_instance(src, tinfo))
            return existing_inst;

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
struct smart_holder_type_caster<std::shared_ptr<T>> : smart_holder_type_caster_load<T>,
                                                      smart_holder_type_caster_class_hooks {
    static constexpr auto name = _<std::shared_ptr<T>>();

    static handle cast(const std::shared_ptr<T> &src, return_value_policy policy, handle parent) {
        if (policy != return_value_policy::automatic
            && policy != return_value_policy::reference_internal) {
            // SMART_HOLDER_WIP: IMPROVABLE: Error message.
            throw cast_error("Invalid return_value_policy for shared_ptr.");
        }

        auto src_raw_ptr = src.get();
        auto st          = type_caster_base<T>::src_and_type(src_raw_ptr);
        if (st.first == nullptr)
            return none().release(); // PyErr was set already.

        void *src_raw_void_ptr         = static_cast<void *>(src_raw_ptr);
        const detail::type_info *tinfo = st.second;
        if (handle existing_inst = find_registered_python_instance(src_raw_void_ptr, tinfo))
            // SMART_HOLDER_WIP: MISSING: Enforcement of consistency with existing smart_holder.
            // SMART_HOLDER_WIP: MISSING: keep_alive.
            return existing_inst;

        auto inst           = reinterpret_steal<object>(make_new_instance(tinfo->type));
        auto *inst_raw_ptr  = reinterpret_cast<instance *>(inst.ptr());
        inst_raw_ptr->owned = true;
        void *&valueptr     = values_and_holders(inst_raw_ptr).begin()->value_ptr();
        valueptr            = src_raw_void_ptr;

        auto smhldr = pybindit::memory::smart_holder::from_shared_ptr(src);
        tinfo->init_instance(inst_raw_ptr, static_cast<const void *>(&smhldr));

        if (policy == return_value_policy::reference_internal)
            keep_alive_impl(inst, parent);

        return inst.release();
    }

    template <typename>
    using cast_op_type = std::shared_ptr<T>;

    operator std::shared_ptr<T>() { return this->loaded_as_shared_ptr(); }
};

template <typename T>
struct smart_holder_type_caster<std::shared_ptr<T const>> : smart_holder_type_caster_load<T>,
                                                            smart_holder_type_caster_class_hooks {
    static constexpr auto name = _<std::shared_ptr<T const>>();

    static handle
    cast(const std::shared_ptr<T const> &src, return_value_policy policy, handle parent) {
        return smart_holder_type_caster<std::shared_ptr<T>>::cast(
            std::const_pointer_cast<T>(src), // Const2Mutbl
            policy,
            parent);
    }

    template <typename>
    using cast_op_type = std::shared_ptr<T const>;

    operator std::shared_ptr<T const>() { return this->loaded_as_shared_ptr(); } // Mutbl2Const
};

template <typename T, typename D>
struct smart_holder_type_caster<std::unique_ptr<T, D>> : smart_holder_type_caster_load<T>,
                                                         smart_holder_type_caster_class_hooks {
    static constexpr auto name = _<std::unique_ptr<T, D>>();

    static handle cast(std::unique_ptr<T, D> &&src, return_value_policy policy, handle parent) {
        if (policy != return_value_policy::automatic
            && policy != return_value_policy::reference_internal
            && policy != return_value_policy::move) {
            // SMART_HOLDER_WIP: IMPROVABLE: Error message.
            throw cast_error("Invalid return_value_policy for unique_ptr.");
        }

        auto src_raw_ptr = src.get();
        auto st          = type_caster_base<T>::src_and_type(src_raw_ptr);
        if (st.first == nullptr)
            return none().release(); // PyErr was set already.

        void *src_raw_void_ptr         = static_cast<void *>(src_raw_ptr);
        const detail::type_info *tinfo = st.second;
        if (find_registered_python_instance(src_raw_void_ptr, tinfo))
            throw cast_error("Invalid unique_ptr: another instance owns this pointer already.");

        auto inst           = reinterpret_steal<object>(make_new_instance(tinfo->type));
        auto *inst_raw_ptr  = reinterpret_cast<instance *>(inst.ptr());
        inst_raw_ptr->owned = true;
        void *&valueptr     = values_and_holders(inst_raw_ptr).begin()->value_ptr();
        valueptr            = src_raw_void_ptr;

        auto smhldr = pybindit::memory::smart_holder::from_unique_ptr(std::move(src));
        tinfo->init_instance(inst_raw_ptr, static_cast<const void *>(&smhldr));

        if (policy == return_value_policy::reference_internal)
            keep_alive_impl(inst, parent);

        return inst.release();
    }

    template <typename>
    using cast_op_type = std::unique_ptr<T, D>;

    operator std::unique_ptr<T, D>() { return this->template loaded_as_unique_ptr<D>(); }
};

template <typename T, typename D>
struct smart_holder_type_caster<std::unique_ptr<T const, D>>
    : smart_holder_type_caster_load<T>, smart_holder_type_caster_class_hooks {
    static constexpr auto name = _<std::unique_ptr<T const, D>>();

    static handle
    cast(std::unique_ptr<T const, D> &&src, return_value_policy policy, handle parent) {
        return smart_holder_type_caster<std::unique_ptr<T, D>>::cast(
            std::unique_ptr<T, D>(const_cast<T *>(src.release())), // Const2Mutbl
            policy,
            parent);
    }

    template <typename>
    using cast_op_type = std::unique_ptr<T const, D>;

    operator std::unique_ptr<T const, D>() { return this->template loaded_as_unique_ptr<D>(); }
};

#ifndef PYBIND11_USE_SMART_HOLDER_AS_DEFAULT

#    define PYBIND11_SMART_HOLDER_TYPE_CASTERS(T)                                                 \
        namespace pybind11 {                                                                      \
        namespace detail {                                                                        \
        template <>                                                                               \
        class type_caster<T> : public smart_holder_type_caster<T> {};                             \
        template <>                                                                               \
        class type_caster<std::shared_ptr<T>>                                                     \
            : public smart_holder_type_caster<std::shared_ptr<T>> {};                             \
        template <>                                                                               \
        class type_caster<std::shared_ptr<T const>>                                               \
            : public smart_holder_type_caster<std::shared_ptr<T const>> {};                       \
        template <typename D>                                                                     \
        class type_caster<std::unique_ptr<T, D>>                                                  \
            : public smart_holder_type_caster<std::unique_ptr<T, D>> {};                          \
        template <typename D>                                                                     \
        class type_caster<std::unique_ptr<T const, D>>                                            \
            : public smart_holder_type_caster<std::unique_ptr<T const, D>> {};                    \
        }                                                                                         \
        }
#else

#    define PYBIND11_SMART_HOLDER_TYPE_CASTERS(T)

template <typename T>
class type_caster_for_class_ : public smart_holder_type_caster<T> {};

template <typename T>
class type_caster_for_class_<std::shared_ptr<T>>
    : public smart_holder_type_caster<std::shared_ptr<T>> {};

template <typename T>
class type_caster_for_class_<std::shared_ptr<T const>>
    : public smart_holder_type_caster<std::shared_ptr<T const>> {};

template <typename T, typename D>
class type_caster_for_class_<std::unique_ptr<T, D>>
    : public smart_holder_type_caster<std::unique_ptr<T, D>> {};

template <typename T, typename D>
class type_caster_for_class_<std::unique_ptr<T const, D>>
    : public smart_holder_type_caster<std::unique_ptr<T const, D>> {};

#endif

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
