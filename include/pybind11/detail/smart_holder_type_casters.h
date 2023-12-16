// Copyright (c) 2021 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include "../gil.h"
#include "../pytypes.h"
#include "../trampoline_self_life_support.h"
#include "common.h"
#include "descr.h"
#include "dynamic_raw_ptr_cast_if_possible.h"
#include "internals.h"
#include "smart_holder_poc.h"
#include "smart_holder_sfinae_hooks_only.h"
#include "type_caster_base.h"
#include "typeid.h"

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
extern "C" inline PyObject *pybind11_object_new(PyTypeObject *type, PyObject *, PyObject *);

// Replace all occurrences of substrings in a string.
inline void replace_all(std::string &str, const std::string &from, const std::string &to) {
    if (str.empty()) {
        return;
    }
    size_t pos = 0;
    while ((pos = str.find(from, pos)) != std::string::npos) {
        str.replace(pos, from.length(), to);
        pos += to.length();
    }
}

inline bool type_is_pybind11_class_(PyTypeObject *type_obj) {
#if defined(PYPY_VERSION)
    auto &internals = get_internals();
    return bool(internals.registered_types_py.find(type_obj)
                != internals.registered_types_py.end());
#else
    return bool(type_obj->tp_new == pybind11_object_new);
#endif
}

inline bool is_instance_method_of_type(PyTypeObject *type_obj, PyObject *attr_name) {
    PyObject *descr = _PyType_Lookup(type_obj, attr_name);
    return bool((descr != nullptr) && PyInstanceMethod_Check(descr));
}

inline object try_get_as_capsule_method(PyObject *obj, PyObject *attr_name) {
    if (PyType_Check(obj)) {
        return object();
    }
    PyTypeObject *type_obj = Py_TYPE(obj);
    bool known_callable = false;
    if (type_is_pybind11_class_(type_obj)) {
        if (!is_instance_method_of_type(type_obj, attr_name)) {
            return object();
        }
        known_callable = true;
    }
    PyObject *method = PyObject_GetAttr(obj, attr_name);
    if (method == nullptr) {
        PyErr_Clear();
        return object();
    }
    if (!known_callable && PyCallable_Check(method) == 0) {
        Py_DECREF(method);
        return object();
    }
    return reinterpret_steal<object>(method);
}

inline void *try_as_void_ptr_capsule_get_pointer(handle src, const char *typeid_name) {
    std::string suffix = clean_type_id(typeid_name);
    replace_all(suffix, "::", "_"); // Convert `a::b::c` to `a_b_c`.
    replace_all(suffix, "*", "");
    object as_capsule_method = try_get_as_capsule_method(src.ptr(), str("as_" + suffix).ptr());
    if (as_capsule_method) {
        object void_ptr_capsule = as_capsule_method();
        if (isinstance<capsule>(void_ptr_capsule)) {
            return reinterpret_borrow<capsule>(void_ptr_capsule).get_pointer();
        }
    }
    return nullptr;
}

// The modified_type_caster_generic_load_impl could replace type_caster_generic::load_impl but not
// vice versa. The main difference is that the original code only propagates a reference to the
// held value, while the modified implementation propagates value_and_holder.
class modified_type_caster_generic_load_impl {
public:
    PYBIND11_NOINLINE explicit modified_type_caster_generic_load_impl(
        const std::type_info &type_info)
        : typeinfo(get_type_info(type_info)), cpptype(&type_info) {}

    explicit modified_type_caster_generic_load_impl(const type_info *typeinfo = nullptr)
        : typeinfo(typeinfo), cpptype(typeinfo ? typeinfo->cpptype : nullptr) {}

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
                const auto *type = v_h.type ? v_h.type : typeinfo;
                if (type->operator_new) {
                    vptr = type->operator_new(type->type_size);
                } else {
#if defined(__cpp_aligned_new) && (!defined(_MSC_VER) || _MSC_VER >= 1912)
                    if (type->type_align > __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
                        vptr = ::operator new(type->type_size, std::align_val_t(type->type_align));
                    } else {
                        vptr = ::operator new(type->type_size);
                    }
#else
                    vptr = ::operator new(type->type_size);
#endif
                }
            }
            // type_caster_generic::load_value END
        }
        loaded_v_h = v_h;
        loaded_v_h.type = typeinfo;
    }

    bool try_implicit_casts(handle src, bool convert) {
        for (const auto &cast : typeinfo->implicit_casts) {
            modified_type_caster_generic_load_impl sub_caster(*cast.first);
            if (sub_caster.load(src, convert)) {
                if (loaded_v_h_cpptype != nullptr) {
                    pybind11_fail("smart_holder_type_casters: try_implicit_casts failure.");
                }
                loaded_v_h = sub_caster.loaded_v_h;
                loaded_v_h_cpptype = cast.first;
                // the sub_caster is being discarded, so steal its vector
                implicit_casts = std::move(sub_caster.implicit_casts);
                implicit_casts.emplace_back(cast.second);
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

    bool try_as_void_ptr_capsule(handle src) {
        unowned_void_ptr_from_void_ptr_capsule
            = try_as_void_ptr_capsule_get_pointer(src, cpptype->name());
        if (unowned_void_ptr_from_void_ptr_capsule != nullptr) {
            return true;
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
        if (!hasattr(pytype, local_key)) {
            return false;
        }

        type_info *foreign_typeinfo = reinterpret_borrow<capsule>(getattr(pytype, local_key));
        // Only consider this foreign loader if actually foreign and is a loader of the correct cpp
        // type
        if (foreign_typeinfo->module_local_load == &local_load
            || (cpptype && !same_type(*cpptype, *foreign_typeinfo->cpptype))) {
            return false;
        }

        void *foreign_loader_void_ptr
            = foreign_typeinfo->module_local_load(src.ptr(), foreign_typeinfo);
        if (foreign_loader_void_ptr != nullptr) {
            auto foreign_loader = std::unique_ptr<modified_type_caster_generic_load_impl>(
                static_cast<modified_type_caster_generic_load_impl *>(foreign_loader_void_ptr));
            // Magic number intentionally hard-coded for simplicity and maximum robustness.
            if (foreign_loader->local_load_safety_guard != 1887406645) {
                pybind11_fail("smart_holder_type_casters: Unexpected local_load_safety_guard,"
                              " possibly due to py::class_ holder mixup.");
            }
            if (loaded_v_h_cpptype != nullptr) {
                pybind11_fail("smart_holder_type_casters: try_load_foreign_module_local failure.");
            }
            loaded_v_h = foreign_loader->loaded_v_h;
            loaded_v_h_cpptype = foreign_loader->loaded_v_h_cpptype;
            // SMART_HOLDER_WIP: should this be a copy or move?
            implicit_casts = foreign_loader->implicit_casts;
            return true;
        }
        return false;
    }

    // Implementation of `load`; this takes the type of `this` so that it can dispatch the relevant
    // bits of code between here and copyable_holder_caster where the two classes need different
    // logic (without having to resort to virtual inheritance).
    template <typename ThisT>
    PYBIND11_NOINLINE bool load_impl(handle src, bool convert) {
        if (!src) {
            return false;
        }
        if (!typeinfo) {
            return try_load_foreign_module_local(src);
        }

        auto &this_ = static_cast<ThisT &>(*this);

        PyTypeObject *srctype = Py_TYPE(src.ptr());

        // Case 1: If src is an exact type match for the target type then we can reinterpret_cast
        // the instance's value pointer to the target type:
        if (srctype == typeinfo->type) {
            this_.load_value_and_holder(
                reinterpret_cast<instance *>(src.ptr())->get_value_and_holder());
            return true;
        }
        // Case 2: We have a derived class
        if (PyType_IsSubtype(srctype, typeinfo->type)) {
            const auto &bases = all_type_info(srctype); // subtype bases
            bool no_cpp_mi = typeinfo->simple_type;

            // Case 2a: the python type is a Python-inherited derived class that inherits from just
            // one simple (no MI) pybind11 class, or is an exact match, so the C++ instance is of
            // the right type and we can use reinterpret_cast.
            // (This is essentially the same as case 2b, but because not using multiple inheritance
            // is extremely common, we handle it specially to avoid the loop iterator and type
            // pointer lookup overhead)
            if (bases.size() == 1 && (no_cpp_mi || bases.front()->type == typeinfo->type)) {
                this_.load_value_and_holder(
                    reinterpret_cast<instance *>(src.ptr())->get_value_and_holder());
                loaded_v_h_cpptype = bases.front()->cpptype;
                reinterpret_cast_deemed_ok = true;
                return true;
            }
            // Case 2b: the python type inherits from multiple C++ bases.  Check the bases to see
            // if we can find an exact match (or, for a simple C++ type, an inherited match); if
            // so, we can safely reinterpret_cast to the relevant pointer.
            if (bases.size() > 1) {
                for (auto *base : bases) {
                    if (no_cpp_mi ? PyType_IsSubtype(base->type, typeinfo->type)
                                  : base->type == typeinfo->type) {
                        this_.load_value_and_holder(
                            reinterpret_cast<instance *>(src.ptr())->get_value_and_holder(base));
                        loaded_v_h_cpptype = base->cpptype;
                        reinterpret_cast_deemed_ok = true;
                        return true;
                    }
                }
            }

            // Case 2c: C++ multiple inheritance is involved and we couldn't find an exact type
            // match in the registered bases, above, so try implicit casting (needed for proper C++
            // casting when MI is involved).
            if (this_.try_implicit_casts(src, convert)) {
                return true;
            }
        }

        // Perform an implicit conversion
        if (convert) {
            for (const auto &converter : typeinfo->implicit_conversions) {
                auto temp = reinterpret_steal<object>(converter(src.ptr(), typeinfo->type));
                if (load_impl<ThisT>(temp, false)) {
                    loader_life_support::add_patient(temp);
                    return true;
                }
            }
            if (this_.try_direct_conversions(src)) {
                return true;
            }
        }

        // Failed to match local typeinfo. Try again with global.
        if (typeinfo->module_local) {
            if (auto *gtype = get_global_type_info(*typeinfo->cpptype)) {
                typeinfo = gtype;
                return load(src, false);
            }
        }

        // Global typeinfo has precedence over foreign module_local
        if (try_load_foreign_module_local(src)) {
            return true;
        }

        if (src.is_none()) {
            // Defer accepting None to other overloads (if we aren't in convert mode):
            if (!convert) {
                return false;
            }
            loaded_v_h = value_and_holder();
            return true;
        }
        if (convert && cpptype && try_as_void_ptr_capsule(src)) {
            return true;
        }
        return false;
    }

    const type_info *typeinfo = nullptr;
    const std::type_info *cpptype = nullptr;
    void *unowned_void_ptr_from_direct_conversion = nullptr;
    void *unowned_void_ptr_from_void_ptr_capsule = nullptr;
    const std::type_info *loaded_v_h_cpptype = nullptr;
    std::vector<void *(*) (void *)> implicit_casts;
    value_and_holder loaded_v_h;
    bool reinterpret_cast_deemed_ok = false;
    // Magic number intentionally hard-coded, to guard against class_ holder mixups.
    // Ideally type_caster_generic would have a similar guard, but this requires a change there.
    // SMART_HOLDER_WIP: If it is decided that this guard is useful long term, potentially
    // set/reset this value in ctor/dtor, mark volatile.
    std::size_t local_load_safety_guard = 1887406645; // 32-bit compatible value for portability.
};

struct smart_holder_type_caster_class_hooks : smart_holder_type_caster_base_tag {
    static decltype(&modified_type_caster_generic_load_impl::local_load)
    get_local_load_function_ptr() {
        return &modified_type_caster_generic_load_impl::local_load;
    }

    using holder_type = pybindit::memory::smart_holder;

    template <typename WrappedType>
    static bool try_initialization_using_shared_from_this(holder_type *, WrappedType *, ...) {
        return false;
    }

    // Adopting existing approach used by type_caster_base, although it leads to somewhat fuzzy
    // ownership semantics: if we detected via shared_from_this that a shared_ptr exists already,
    // it is reused, irrespective of the return_value_policy in effect.
    // "SomeBaseOfWrappedType" is needed because std::enable_shared_from_this is not necessarily a
    // direct base of WrappedType.
    template <typename WrappedType, typename SomeBaseOfWrappedType>
    static bool try_initialization_using_shared_from_this(
        holder_type *uninitialized_location,
        WrappedType *value_ptr_w_t,
        const std::enable_shared_from_this<SomeBaseOfWrappedType> *) {
        auto shd_ptr = std::dynamic_pointer_cast<WrappedType>(
            detail::try_get_shared_from_this(value_ptr_w_t));
        if (!shd_ptr) {
            return false;
        }
        // Note: inst->owned ignored.
        new (uninitialized_location) holder_type(holder_type::from_shared_ptr(shd_ptr));
        return true;
    }

    template <typename WrappedType, typename AliasType>
    static void init_instance_for_type(detail::instance *inst, const void *holder_const_void_ptr) {
        // Need for const_cast is a consequence of the type_info::init_instance type:
        // void (*init_instance)(instance *, const void *);
        auto *holder_void_ptr = const_cast<void *>(holder_const_void_ptr);

        auto v_h = inst->get_value_and_holder(detail::get_type_info(typeid(WrappedType)));
        if (!v_h.instance_registered()) {
            register_instance(inst, v_h.value_ptr(), v_h.type);
            v_h.set_instance_registered();
        }
        auto *uninitialized_location = std::addressof(v_h.holder<holder_type>());
        auto *value_ptr_w_t = v_h.value_ptr<WrappedType>();
        bool pointee_depends_on_holder_owner
            = dynamic_raw_ptr_cast_if_possible<AliasType>(value_ptr_w_t) != nullptr;
        if (holder_void_ptr) {
            // Note: inst->owned ignored.
            auto *holder_ptr = static_cast<holder_type *>(holder_void_ptr);
            new (uninitialized_location) holder_type(std::move(*holder_ptr));
        } else if (!try_initialization_using_shared_from_this(
                       uninitialized_location, value_ptr_w_t, value_ptr_w_t)) {
            if (inst->owned) {
                new (uninitialized_location) holder_type(holder_type::from_raw_ptr_take_ownership(
                    value_ptr_w_t, /*void_cast_raw_ptr*/ pointee_depends_on_holder_owner));
            } else {
                new (uninitialized_location)
                    holder_type(holder_type::from_raw_ptr_unowned(value_ptr_w_t));
            }
        }
        v_h.holder<holder_type>().pointee_depends_on_holder_owner
            = pointee_depends_on_holder_owner;
        v_h.set_holder_constructed();
    }

    template <typename T, typename D>
    static smart_holder smart_holder_from_unique_ptr(std::unique_ptr<T, D> &&unq_ptr,
                                                     bool void_cast_raw_ptr) {
        void *void_ptr = void_cast_raw_ptr ? static_cast<void *>(unq_ptr.get()) : nullptr;
        return pybindit::memory::smart_holder::from_unique_ptr(std::move(unq_ptr), void_ptr);
    }

    template <typename T>
    static smart_holder smart_holder_from_shared_ptr(std::shared_ptr<T> shd_ptr) {
        return pybindit::memory::smart_holder::from_shared_ptr(shd_ptr);
    }
};

struct shared_ptr_parent_life_support {
    PyObject *parent;
    explicit shared_ptr_parent_life_support(PyObject *parent) : parent{parent} {
        Py_INCREF(parent);
    }
    // NOLINTNEXTLINE(readability-make-member-function-const)
    void operator()(void *) {
        gil_scoped_acquire gil;
        Py_DECREF(parent);
    }
};

struct shared_ptr_trampoline_self_life_support {
    PyObject *self;
    explicit shared_ptr_trampoline_self_life_support(instance *inst)
        : self{reinterpret_cast<PyObject *>(inst)} {
        gil_scoped_acquire gil;
        Py_INCREF(self);
    }
    // NOLINTNEXTLINE(readability-make-member-function-const)
    void operator()(void *) {
        gil_scoped_acquire gil;
        Py_DECREF(self);
    }
};

template <typename T,
          typename D,
          typename std::enable_if<std::is_default_constructible<D>::value, int>::type = 0>
inline std::unique_ptr<T, D> unique_with_deleter(T *raw_ptr, std::unique_ptr<D> &&deleter) {
    if (deleter == nullptr) {
        return std::unique_ptr<T, D>(raw_ptr);
    }
    return std::unique_ptr<T, D>(raw_ptr, std::move(*deleter));
}

template <typename T,
          typename D,
          typename std::enable_if<!std::is_default_constructible<D>::value, int>::type = 0>
inline std::unique_ptr<T, D> unique_with_deleter(T *raw_ptr, std::unique_ptr<D> &&deleter) {
    if (deleter == nullptr) {
        pybind11_fail("smart_holder_type_casters: deleter is not default constructible and no"
                      " instance available to return.");
    }
    return std::unique_ptr<T, D>(raw_ptr, std::move(*deleter));
}

template <typename T>
struct smart_holder_type_caster_load {
    using holder_type = pybindit::memory::smart_holder;

    bool load(handle src, bool convert) {
        static_assert(type_uses_smart_holder_type_caster<T>::value, "Internal consistency error.");
        load_impl = modified_type_caster_generic_load_impl(typeid(T));
        if (!load_impl.load(src, convert)) {
            return false;
        }
        return true;
    }

    T *loaded_as_raw_ptr_unowned() const {
        void *void_ptr = load_impl.unowned_void_ptr_from_void_ptr_capsule;
        if (void_ptr == nullptr) {
            void_ptr = load_impl.unowned_void_ptr_from_direct_conversion;
        }
        if (void_ptr == nullptr) {
            if (have_holder()) {
                throw_if_uninitialized_or_disowned_holder(typeid(T));
                void_ptr = holder().template as_raw_ptr_unowned<void>();
            } else if (load_impl.loaded_v_h.vh != nullptr) {
                void_ptr = load_impl.loaded_v_h.value_ptr();
            }
            if (void_ptr == nullptr) {
                return nullptr;
            }
        }
        return convert_type(void_ptr);
    }

    T &loaded_as_lvalue_ref() const {
        T *raw_ptr = loaded_as_raw_ptr_unowned();
        if (raw_ptr == nullptr) {
            throw reference_cast_error();
        }
        return *raw_ptr;
    }

    std::shared_ptr<T> make_shared_ptr_with_responsible_parent(handle parent) const {
        return std::shared_ptr<T>(loaded_as_raw_ptr_unowned(),
                                  shared_ptr_parent_life_support(parent.ptr()));
    }

    std::shared_ptr<T> loaded_as_shared_ptr(handle responsible_parent = nullptr) const {
        if (load_impl.unowned_void_ptr_from_void_ptr_capsule) {
            if (responsible_parent) {
                return make_shared_ptr_with_responsible_parent(responsible_parent);
            }
            throw cast_error("Unowned pointer from a void pointer capsule cannot be converted to a"
                             " std::shared_ptr.");
        }
        if (load_impl.unowned_void_ptr_from_direct_conversion != nullptr) {
            if (responsible_parent) {
                return make_shared_ptr_with_responsible_parent(responsible_parent);
            }
            throw cast_error("Unowned pointer from direct conversion cannot be converted to a"
                             " std::shared_ptr.");
        }
        if (!have_holder()) {
            return nullptr;
        }
        throw_if_uninitialized_or_disowned_holder(typeid(T));
        holder_type &hld = holder();
        hld.ensure_is_not_disowned("loaded_as_shared_ptr");
        if (hld.vptr_is_using_noop_deleter) {
            if (responsible_parent) {
                return make_shared_ptr_with_responsible_parent(responsible_parent);
            }
            throw std::runtime_error("Non-owning holder (loaded_as_shared_ptr).");
        }
        auto *void_raw_ptr = hld.template as_raw_ptr_unowned<void>();
        auto *type_raw_ptr = convert_type(void_raw_ptr);
        if (hld.pointee_depends_on_holder_owner) {
            auto *vptr_gd_ptr = std::get_deleter<pybindit::memory::guarded_delete>(hld.vptr);
            if (vptr_gd_ptr != nullptr) {
                std::shared_ptr<void> released_ptr = vptr_gd_ptr->released_ptr.lock();
                if (released_ptr) {
                    return std::shared_ptr<T>(released_ptr, type_raw_ptr);
                }
                std::shared_ptr<T> to_be_released(
                    type_raw_ptr,
                    shared_ptr_trampoline_self_life_support(load_impl.loaded_v_h.inst));
                vptr_gd_ptr->released_ptr = to_be_released;
                return to_be_released;
            }
            auto *sptsls_ptr = std::get_deleter<shared_ptr_trampoline_self_life_support>(hld.vptr);
            if (sptsls_ptr != nullptr) {
                // This code is reachable only if there are multiple registered_instances for the
                // same pointee.
                if (reinterpret_cast<PyObject *>(load_impl.loaded_v_h.inst) == sptsls_ptr->self) {
                    pybind11_fail("smart_holder_type_casters loaded_as_shared_ptr failure: "
                                  "load_impl.loaded_v_h.inst == sptsls_ptr->self");
                }
            }
            if (sptsls_ptr != nullptr
                || !pybindit::memory::type_has_shared_from_this(type_raw_ptr)) {
                return std::shared_ptr<T>(
                    type_raw_ptr,
                    shared_ptr_trampoline_self_life_support(load_impl.loaded_v_h.inst));
            }
            if (hld.vptr_is_external_shared_ptr) {
                pybind11_fail("smart_holder_type_casters loaded_as_shared_ptr failure: not "
                              "implemented: trampoline-self-life-support for external shared_ptr "
                              "to type inheriting from std::enable_shared_from_this.");
            }
            pybind11_fail("smart_holder_type_casters: loaded_as_shared_ptr failure: internal "
                          "inconsistency.");
        }
        std::shared_ptr<void> void_shd_ptr = hld.template as_shared_ptr<void>();
        return std::shared_ptr<T>(void_shd_ptr, type_raw_ptr);
    }

    template <typename D>
    std::unique_ptr<T, D> loaded_as_unique_ptr(const char *context = "loaded_as_unique_ptr") {
        if (load_impl.unowned_void_ptr_from_void_ptr_capsule) {
            throw cast_error("Unowned pointer from a void pointer capsule cannot be converted to a"
                             " std::unique_ptr.");
        }
        if (load_impl.unowned_void_ptr_from_direct_conversion != nullptr) {
            throw cast_error("Unowned pointer from direct conversion cannot be converted to a"
                             " std::unique_ptr.");
        }
        if (!have_holder()) {
            return unique_with_deleter<T, D>(nullptr, std::unique_ptr<D>());
        }
        throw_if_uninitialized_or_disowned_holder(typeid(T));
        throw_if_instance_is_currently_owned_by_shared_ptr();
        holder().ensure_is_not_disowned(context);
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
            = dynamic_raw_ptr_cast_if_possible<trampoline_self_life_support>(raw_type_ptr);
        if (self_life_support == nullptr && holder().pointee_depends_on_holder_owner) {
            throw value_error("Alias class (also known as trampoline) does not inherit from "
                              "py::trampoline_self_life_support, therefore the ownership of this "
                              "instance cannot safely be transferred to C++.");
        }

        // Temporary variable to store the extracted deleter in.
        std::unique_ptr<D> extracted_deleter;

        auto *gd = std::get_deleter<pybindit::memory::guarded_delete>(holder().vptr);
        if (gd && gd->use_del_fun) { // Note the ensure_compatible_rtti_uqp_del<T, D>() call above.
            // In smart_holder_poc, a custom  deleter is always stored in a guarded delete.
            // The guarded delete's std::function<void(void*)> actually points at the
            // custom_deleter type, so we can verify it is of the custom deleter type and
            // finally extract its deleter.
            using custom_deleter_D = pybindit::memory::custom_deleter<T, D>;
            const auto &custom_deleter_ptr = gd->del_fun.template target<custom_deleter_D>();
            assert(custom_deleter_ptr != nullptr);
            // Now that we have confirmed the type of the deleter matches the desired return
            // value we can extract the function.
            extracted_deleter = std::unique_ptr<D>(new D(std::move(custom_deleter_ptr->deleter)));
        }

        // Critical transfer-of-ownership section. This must stay together.
        if (self_life_support != nullptr) {
            holder().disown();
        } else {
            holder().release_ownership();
        }
        auto result = unique_with_deleter<T, D>(raw_type_ptr, std::move(extracted_deleter));
        if (self_life_support != nullptr) {
            self_life_support->activate_life_support(load_impl.loaded_v_h);
        } else {
            load_impl.loaded_v_h.value_ptr() = nullptr;
            deregister_instance(
                load_impl.loaded_v_h.inst, value_void_ptr, load_impl.loaded_v_h.type);
        }
        // Critical section end.

        return result;
    }

    // This function will succeed even if the `responsible_parent` does not own the
    // wrapped C++ object directly.
    // It is the responsibility of the caller to ensure that the `responsible_parent`
    // has a `keep_alive` relationship with the owner of the wrapped C++ object, or
    // that the wrapped C++ object lives for the duration of the process.
    static std::shared_ptr<T> shared_ptr_from_python(handle responsible_parent) {
        smart_holder_type_caster_load<T> loader;
        loader.load(responsible_parent, false);
        return loader.loaded_as_shared_ptr(responsible_parent);
    }

private:
    modified_type_caster_generic_load_impl load_impl;

    bool have_holder() const {
        return load_impl.loaded_v_h.vh != nullptr && load_impl.loaded_v_h.holder_constructed();
    }

    holder_type &holder() const { return load_impl.loaded_v_h.holder<holder_type>(); }

    // have_holder() must be true or this function will fail.
    void throw_if_uninitialized_or_disowned_holder(const char *typeid_name) const {
        static const std::string missing_value_msg = "Missing value for wrapped C++ type `";
        if (!holder().is_populated) {
            throw value_error(missing_value_msg + clean_type_id(typeid_name)
                              + "`: Python instance is uninitialized.");
        }
        if (!holder().has_pointee()) {
            throw value_error(missing_value_msg + clean_type_id(typeid_name)
                              + "`: Python instance was disowned.");
        }
    }

    void throw_if_uninitialized_or_disowned_holder(const std::type_info &type_info) const {
        throw_if_uninitialized_or_disowned_holder(type_info.name());
    }

    // have_holder() must be true or this function will fail.
    void throw_if_instance_is_currently_owned_by_shared_ptr() const {
        auto vptr_gd_ptr = std::get_deleter<pybindit::memory::guarded_delete>(holder().vptr);
        if (vptr_gd_ptr != nullptr && !vptr_gd_ptr->released_ptr.expired()) {
            throw value_error("Python instance is currently owned by a std::shared_ptr.");
        }
    }

    T *convert_type(void *void_ptr) const {
        if (void_ptr != nullptr && load_impl.loaded_v_h_cpptype != nullptr
            && !load_impl.reinterpret_cast_deemed_ok && !load_impl.implicit_casts.empty()) {
            for (auto implicit_cast : load_impl.implicit_casts) {
                void_ptr = implicit_cast(void_ptr);
            }
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
    static constexpr auto name = const_name<T>();

    // static handle cast(T, ...)
    // is redundant (leads to ambiguous overloads).

    static handle cast(T &&src, return_value_policy /*policy*/, handle parent) {
        // type_caster_base BEGIN
        return cast(&src, return_value_policy::move, parent);
        // type_caster_base END
    }

    static handle cast(T const &src, return_value_policy policy, handle parent) {
        // type_caster_base BEGIN
        if (policy == return_value_policy::automatic
            || policy == return_value_policy::automatic_reference
            || policy == return_value_policy::_clif_automatic) {
            policy = return_value_policy::copy;
        }
        return cast(&src, policy, parent);
        // type_caster_base END
    }

    static handle cast(T &src, return_value_policy policy, handle parent) {
        if (policy == return_value_policy::_clif_automatic) {
            if (is_move_constructible<T>::value) {
                policy = return_value_policy::move;
            } else {
                policy = return_value_policy::automatic;
            }
        }
        return cast(const_cast<T const &>(src), policy, parent); // Mutbl2Const
    }

    static handle cast(T const *src, return_value_policy policy, handle parent) {
        auto st = type_caster_base<T>::src_and_type(src);
        if (policy == return_value_policy::_clif_automatic) {
            policy = return_value_policy::copy;
        }
        return cast_const_raw_ptr( // Originally type_caster_generic::cast.
            st.first,
            policy,
            parent,
            st.second,
            make_constructor::make_copy_constructor(src),
            make_constructor::make_move_constructor(src));
    }

    static handle cast(T *src, return_value_policy policy, handle parent) {
        if (policy == return_value_policy::_clif_automatic) {
            if (parent) {
                policy = return_value_policy::reference_internal;
            } else {
                policy = return_value_policy::reference;
            }
        }
        return cast(const_cast<T const *>(src), policy, parent); // Mutbl2Const
    }

    template <typename T_>
    using cast_op_type = conditional_t<
        std::is_same<remove_reference_t<T_>, T const *>::value,
        T const *,
        conditional_t<std::is_same<remove_reference_t<T_>, T *>::value,
                      T *,
                      conditional_t<std::is_same<T_, T const &>::value, T const &, T &>>>;

    // The const operators here prove that the existing type_caster mechanism already supports
    // const-correctness. However, fully implementing const-correctness inside this type_caster
    // is still a major project.
    // NOLINTNEXTLINE(google-explicit-constructor)
    operator T const &() const {
        return const_cast<smart_holder_type_caster *>(this)->loaded_as_lvalue_ref();
    }
    // NOLINTNEXTLINE(google-explicit-constructor)
    operator T const *() const {
        return const_cast<smart_holder_type_caster *>(this)->loaded_as_raw_ptr_unowned();
    }
    // NOLINTNEXTLINE(google-explicit-constructor)
    operator T &() { return this->loaded_as_lvalue_ref(); }
    // NOLINTNEXTLINE(google-explicit-constructor)
    operator T *() { return this->loaded_as_raw_ptr_unowned(); }

    // Originally type_caster_generic::cast.
    PYBIND11_NOINLINE static handle cast_const_raw_ptr(const void *_src,
                                                       return_value_policy policy,
                                                       handle parent,
                                                       const detail::type_info *tinfo,
                                                       void *(*copy_constructor)(const void *),
                                                       void *(*move_constructor)(const void *),
                                                       const void *existing_holder = nullptr) {
        if (!tinfo) { // no type info: error will be set already
            return handle();
        }

        void *src = const_cast<void *>(_src);
        if (src == nullptr) {
            return none().release();
        }

        if (handle existing_inst = find_registered_python_instance(src, tinfo)) {
            return existing_inst;
        }

        auto inst = reinterpret_steal<object>(make_new_instance(tinfo->type));
        auto *wrapper = reinterpret_cast<instance *>(inst.ptr());
        wrapper->owned = false;
        void *&valueptr = values_and_holders(wrapper).begin()->value_ptr();

        switch (policy) {
            case return_value_policy::automatic:
            case return_value_policy::take_ownership:
                valueptr = src;
                wrapper->owned = true;
                break;

            case return_value_policy::automatic_reference:
            case return_value_policy::reference:
                valueptr = src;
                wrapper->owned = false;
                break;

            case return_value_policy::copy:
                if (copy_constructor) {
                    valueptr = copy_constructor(src);
                } else {
#if !defined(PYBIND11_DETAILED_ERROR_MESSAGES)
                    throw cast_error("return_value_policy = copy, but type is "
                                     "non-copyable! (#define PYBIND11_DETAILED_ERROR_MESSAGES or "
                                     "compile in debug mode for details)");
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
                if (move_constructor) {
                    valueptr = move_constructor(src);
                } else if (copy_constructor) {
                    valueptr = copy_constructor(src);
                } else {
#if !defined(PYBIND11_DETAILED_ERROR_MESSAGES)
                    throw cast_error("return_value_policy = move, but type is neither "
                                     "movable nor copyable! "
                                     "(#define PYBIND11_DETAILED_ERROR_MESSAGES or compile in "
                                     "debug mode for details)");
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
                valueptr = src;
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
    static constexpr auto name = const_name<T>();

    static handle cast(const std::shared_ptr<T> &src, return_value_policy policy, handle parent) {
        switch (policy) {
            case return_value_policy::automatic:
            case return_value_policy::automatic_reference:
                break;
            case return_value_policy::take_ownership:
                throw cast_error("Invalid return_value_policy for shared_ptr (take_ownership).");
                break;
            case return_value_policy::copy:
            case return_value_policy::move:
                break;
            case return_value_policy::reference:
                throw cast_error("Invalid return_value_policy for shared_ptr (reference).");
                break;
            case return_value_policy::reference_internal:
            case return_value_policy::_return_as_bytes:
            case return_value_policy::_clif_automatic:
                break;
        }
        if (!src) {
            return none().release();
        }

        auto src_raw_ptr = src.get();
        auto st = type_caster_base<T>::src_and_type(src_raw_ptr);
        if (st.second == nullptr) {
            return handle(); // no type info: error will be set already
        }

        void *src_raw_void_ptr = static_cast<void *>(src_raw_ptr);
        const detail::type_info *tinfo = st.second;
        if (handle existing_inst = find_registered_python_instance(src_raw_void_ptr, tinfo)) {
            // SMART_HOLDER_WIP: MISSING: Enforcement of consistency with existing smart_holder.
            // SMART_HOLDER_WIP: MISSING: keep_alive.
            return existing_inst;
        }

        auto inst = reinterpret_steal<object>(make_new_instance(tinfo->type));
        auto *inst_raw_ptr = reinterpret_cast<instance *>(inst.ptr());
        inst_raw_ptr->owned = true;
        void *&valueptr = values_and_holders(inst_raw_ptr).begin()->value_ptr();
        valueptr = src_raw_void_ptr;

        auto smhldr = pybindit::memory::smart_holder::from_shared_ptr(
            std::shared_ptr<void>(src, const_cast<void *>(st.first)));
        tinfo->init_instance(inst_raw_ptr, static_cast<const void *>(&smhldr));

        if (policy == return_value_policy::reference_internal) {
            keep_alive_impl(inst, parent);
        }

        return inst.release();
    }

    template <typename>
    using cast_op_type = std::shared_ptr<T>;

    // NOLINTNEXTLINE(google-explicit-constructor)
    operator std::shared_ptr<T>() { return this->loaded_as_shared_ptr(); }
};

template <typename T>
struct smart_holder_type_caster<std::shared_ptr<T const>> : smart_holder_type_caster_load<T>,
                                                            smart_holder_type_caster_class_hooks {
    static constexpr auto name = const_name<T>();

    static handle
    cast(const std::shared_ptr<T const> &src, return_value_policy policy, handle parent) {
        return smart_holder_type_caster<std::shared_ptr<T>>::cast(
            std::const_pointer_cast<T>(src), // Const2Mutbl
            policy,
            parent);
    }

    template <typename>
    using cast_op_type = std::shared_ptr<T const>;

    // NOLINTNEXTLINE(google-explicit-constructor)
    operator std::shared_ptr<T const>() { return this->loaded_as_shared_ptr(); } // Mutbl2Const
};

template <typename T, typename D>
struct smart_holder_type_caster<std::unique_ptr<T, D>> : smart_holder_type_caster_load<T>,
                                                         smart_holder_type_caster_class_hooks {
    static constexpr auto name = const_name<T>();

    static handle cast(std::unique_ptr<T, D> &&src, return_value_policy policy, handle parent) {
        if (policy != return_value_policy::automatic
            && policy != return_value_policy::automatic_reference
            && policy != return_value_policy::reference_internal
            && policy != return_value_policy::move
            && policy != return_value_policy::_clif_automatic) {
            // SMART_HOLDER_WIP: IMPROVABLE: Error message.
            throw cast_error("Invalid return_value_policy for unique_ptr.");
        }
        if (!src) {
            return none().release();
        }

        auto st = type_caster_base<T>::src_and_type(src.get());
        if (st.second == nullptr) {
            return handle(); // no type info: error will be set already
        }

        void *src_raw_void_ptr = const_cast<void *>(st.first);
        const detail::type_info *tinfo = st.second;
        if (handle existing_inst = find_registered_python_instance(src_raw_void_ptr, tinfo)) {
            auto *self_life_support
                = dynamic_raw_ptr_cast_if_possible<trampoline_self_life_support>(src.get());
            if (self_life_support != nullptr) {
                value_and_holder &v_h = self_life_support->v_h;
                if (v_h.inst != nullptr && v_h.vh != nullptr) {
                    auto &holder = v_h.holder<pybindit::memory::smart_holder>();
                    if (!holder.is_disowned) {
                        pybind11_fail("smart_holder_type_casters: unexpected "
                                      "smart_holder.is_disowned failure.");
                    }
                    // Critical transfer-of-ownership section. This must stay together.
                    self_life_support->deactivate_life_support();
                    holder.reclaim_disowned();
                    (void) src.release();
                    // Critical section end.
                    return existing_inst;
                }
            }
            throw cast_error("Invalid unique_ptr: another instance owns this pointer already.");
        }

        auto inst = reinterpret_steal<object>(make_new_instance(tinfo->type));
        auto *inst_raw_ptr = reinterpret_cast<instance *>(inst.ptr());
        inst_raw_ptr->owned = true;
        void *&valueptr = values_and_holders(inst_raw_ptr).begin()->value_ptr();
        valueptr = src_raw_void_ptr;

        if (static_cast<void *>(src.get()) == src_raw_void_ptr) {
            // This is a multiple-inheritance situation that is incompatible with the current
            // shared_from_this handling (see PR #3023).
            // SMART_HOLDER_WIP: IMPROVABLE: Is there a better solution?
            src_raw_void_ptr = nullptr;
        }
        auto smhldr
            = pybindit::memory::smart_holder::from_unique_ptr(std::move(src), src_raw_void_ptr);
        tinfo->init_instance(inst_raw_ptr, static_cast<const void *>(&smhldr));

        if (policy == return_value_policy::reference_internal) {
            keep_alive_impl(inst, parent);
        }

        return inst.release();
    }
    static handle
    cast(const std::unique_ptr<T, D> &src, return_value_policy policy, handle parent) {
        if (!src) {
            return none().release();
        }
        if (policy == return_value_policy::automatic) {
            policy = return_value_policy::reference_internal;
        }
        if (policy != return_value_policy::reference_internal) {
            throw cast_error("Invalid return_value_policy for unique_ptr&");
        }
        return smart_holder_type_caster<T>::cast(src.get(), policy, parent);
    }

    template <typename>
    using cast_op_type = std::unique_ptr<T, D>;

    // NOLINTNEXTLINE(google-explicit-constructor)
    operator std::unique_ptr<T, D>() { return this->template loaded_as_unique_ptr<D>(); }
};

template <typename T, typename D>
struct smart_holder_type_caster<std::unique_ptr<T const, D>>
    : smart_holder_type_caster_load<T>, smart_holder_type_caster_class_hooks {
    static constexpr auto name = const_name<T>();

    static handle
    cast(std::unique_ptr<T const, D> &&src, return_value_policy policy, handle parent) {
        return smart_holder_type_caster<std::unique_ptr<T, D>>::cast(
            std::unique_ptr<T, D>(const_cast<T *>(src.release()),
                                  std::move(src.get_deleter())), // Const2Mutbl
            policy,
            parent);
    }

    template <typename>
    using cast_op_type = std::unique_ptr<T const, D>;

    // NOLINTNEXTLINE(google-explicit-constructor)
    operator std::unique_ptr<T const, D>() { return this->template loaded_as_unique_ptr<D>(); }
};

#ifndef PYBIND11_USE_SMART_HOLDER_AS_DEFAULT

#    define PYBIND11_SMART_HOLDER_TYPE_CASTERS(...)                                               \
        namespace pybind11 {                                                                      \
        namespace detail {                                                                        \
        template <>                                                                               \
        class type_caster<__VA_ARGS__> : public smart_holder_type_caster<__VA_ARGS__> {};         \
        template <>                                                                               \
        class type_caster<std::shared_ptr<__VA_ARGS__>>                                           \
            : public smart_holder_type_caster<std::shared_ptr<__VA_ARGS__>> {};                   \
        template <>                                                                               \
        class type_caster<std::shared_ptr<__VA_ARGS__ const>>                                     \
            : public smart_holder_type_caster<std::shared_ptr<__VA_ARGS__ const>> {};             \
        template <typename D>                                                                     \
        class type_caster<std::unique_ptr<__VA_ARGS__, D>>                                        \
            : public smart_holder_type_caster<std::unique_ptr<__VA_ARGS__, D>> {};                \
        template <typename D>                                                                     \
        class type_caster<std::unique_ptr<__VA_ARGS__ const, D>>                                  \
            : public smart_holder_type_caster<std::unique_ptr<__VA_ARGS__ const, D>> {};          \
        }                                                                                         \
        }
#else

#    define PYBIND11_SMART_HOLDER_TYPE_CASTERS(...)

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
