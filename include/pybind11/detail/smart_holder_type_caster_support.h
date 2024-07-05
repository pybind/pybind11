#pragma once

#include "../gil.h"
#include "../pytypes.h"
#include "../trampoline_self_life_support.h"
#include "common.h"
#include "dynamic_raw_ptr_cast_if_possible.h"
#include "internals.h"
#include "type_caster_base.h"
#include "typeid.h"

#include <cstdint>
#include <iterator>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)
PYBIND11_NAMESPACE_BEGIN(smart_holder_type_caster_support)

template <typename T, typename D>
handle smart_holder_from_unique_ptr(std::unique_ptr<T, D> &&src,
                                    return_value_policy policy,
                                    handle parent,
                                    const std::pair<const void *, const type_info *> &st) {
    if (policy != return_value_policy::automatic
        && policy != return_value_policy::automatic_reference
        && policy != return_value_policy::reference_internal
        && policy != return_value_policy::move) {
        // SMART_HOLDER_WIP: IMPROVABLE: Error message.
        throw cast_error("Invalid return_value_policy for unique_ptr.");
    }
    if (!src) {
        return none().release();
    }
    void *src_raw_void_ptr = const_cast<void *>(st.first);
    assert(st.second != nullptr);
    const detail::type_info *tinfo = st.second;
    if (handle existing_inst = find_registered_python_instance(src_raw_void_ptr, tinfo)) {
        auto *self_life_support
            = dynamic_raw_ptr_cast_if_possible<trampoline_self_life_support>(src.get());
        if (self_life_support != nullptr) {
            value_and_holder &v_h = self_life_support->v_h;
            if (v_h.inst != nullptr && v_h.vh != nullptr) {
                auto &holder = v_h.holder<pybindit::memory::smart_holder>();
                if (!holder.is_disowned) {
                    pybind11_fail("smart_holder_from_unique_ptr: unexpected "
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

template <typename T, typename D>
handle smart_holder_from_unique_ptr(std::unique_ptr<T const, D> &&src,
                                    return_value_policy policy,
                                    handle parent,
                                    const std::pair<const void *, const type_info *> &st) {
    return smart_holder_from_unique_ptr(
        std::unique_ptr<T, D>(const_cast<T *>(src.release()),
                              std::move(src.get_deleter())), // Const2Mutbl
        policy,
        parent,
        st);
}

template <typename T, typename D>
handle
unique_ptr_to_python(std::unique_ptr<T, D> &&unq_ptr, return_value_policy policy, handle parent) {
    auto *src = unq_ptr.get();
    auto st = type_caster_base<T>::src_and_type(src);
    if (st.second == nullptr) {
        return handle(); // no type info: error will be set already
    }
    if (st.second->default_holder) {
        return smart_holder_from_unique_ptr(std::move(unq_ptr), policy, parent, st);
    }
    return type_caster_generic::cast(st.first,
                                     return_value_policy::take_ownership,
                                     {},
                                     st.second,
                                     nullptr,
                                     nullptr,
                                     std::addressof(unq_ptr));
}

template <typename T>
handle smart_holder_from_shared_ptr(const std::shared_ptr<T> &src,
                                    return_value_policy policy,
                                    handle parent,
                                    const std::pair<const void *, const type_info *> &st) {
    switch (policy) {
        case return_value_policy::automatic:
        case return_value_policy::automatic_reference:
            break;
        case return_value_policy::take_ownership:
            throw cast_error("Invalid return_value_policy for shared_ptr (take_ownership).");
        case return_value_policy::copy:
        case return_value_policy::move:
            break;
        case return_value_policy::reference:
            throw cast_error("Invalid return_value_policy for shared_ptr (reference).");
        case return_value_policy::reference_internal:
            break;
    }
    if (!src) {
        return none().release();
    }

    auto src_raw_ptr = src.get();
    assert(st.second != nullptr);
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

template <typename T>
handle smart_holder_from_shared_ptr(const std::shared_ptr<T const> &src,
                                    return_value_policy policy,
                                    handle parent,
                                    const std::pair<const void *, const type_info *> &st) {
    return smart_holder_from_shared_ptr(std::const_pointer_cast<T>(src), // Const2Mutbl
                                        policy,
                                        parent,
                                        st);
}

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
struct load_helper {
    using holder_type = pybindit::memory::smart_holder;

    value_and_holder loaded_v_h;

    T *loaded_as_raw_ptr_unowned() const {
        void *void_ptr = nullptr;
        if (have_holder()) {
            throw_if_uninitialized_or_disowned_holder(typeid(T));
            void_ptr = holder().template as_raw_ptr_unowned<void>();
        } else if (loaded_v_h.vh != nullptr) {
            void_ptr = loaded_v_h.value_ptr();
        }
        if (void_ptr == nullptr) {
            return nullptr;
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
                    type_raw_ptr, shared_ptr_trampoline_self_life_support(loaded_v_h.inst));
                vptr_gd_ptr->released_ptr = to_be_released;
                return to_be_released;
            }
            auto *sptsls_ptr = std::get_deleter<shared_ptr_trampoline_self_life_support>(hld.vptr);
            if (sptsls_ptr != nullptr) {
                // This code is reachable only if there are multiple registered_instances for the
                // same pointee.
                if (reinterpret_cast<PyObject *>(loaded_v_h.inst) == sptsls_ptr->self) {
                    pybind11_fail(
                        "ssmart_holder_type_caster_support loaded_as_shared_ptr failure: "
                        "loaded_v_h.inst == sptsls_ptr->self");
                }
            }
            if (sptsls_ptr != nullptr
                || !pybindit::memory::type_has_shared_from_this(type_raw_ptr)) {
                return std::shared_ptr<T>(
                    type_raw_ptr, shared_ptr_trampoline_self_life_support(loaded_v_h.inst));
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
        if (!have_holder()) {
            return unique_with_deleter<T, D>(nullptr, std::unique_ptr<D>());
        }
        throw_if_uninitialized_or_disowned_holder(typeid(T));
        throw_if_instance_is_currently_owned_by_shared_ptr();
        holder().ensure_is_not_disowned(context);
        holder().template ensure_compatible_rtti_uqp_del<T, D>(context);
        holder().ensure_use_count_1(context);
        auto raw_void_ptr = holder().template as_raw_ptr_unowned<void>();

        void *value_void_ptr = loaded_v_h.value_ptr();
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
            self_life_support->activate_life_support(loaded_v_h);
        } else {
            loaded_v_h.value_ptr() = nullptr;
            deregister_instance(loaded_v_h.inst, value_void_ptr, loaded_v_h.type);
        }
        // Critical section end.

        return result;
    }

#ifdef BAKEIN_WIP // Is this needed? shared_ptr_from_python(responsible_parent)
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
#endif

private:
    bool have_holder() const {
        return loaded_v_h.vh != nullptr && loaded_v_h.holder_constructed();
    }

    holder_type &holder() const { return loaded_v_h.holder<holder_type>(); }

    // BAKEIN_WIP: This needs to be factored out: see type_caster_base.h
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
#ifdef BAKEIN_WIP // Is this needed? implicit_casts
        if (void_ptr != nullptr && load_impl.loaded_v_h_cpptype != nullptr
            && !load_impl.reinterpret_cast_deemed_ok && !load_impl.implicit_casts.empty()) {
            for (auto implicit_cast : load_impl.implicit_casts) {
                void_ptr = implicit_cast(void_ptr);
            }
        }
#endif
        return static_cast<T *>(void_ptr);
    }
};

PYBIND11_NAMESPACE_END(smart_holder_type_caster_support)
PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
