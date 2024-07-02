#pragma once

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

PYBIND11_NAMESPACE_END(smart_holder_type_caster_support)
PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)