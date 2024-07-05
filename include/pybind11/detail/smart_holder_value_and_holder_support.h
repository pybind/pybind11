#pragma once

// BAKEIN_WIP: IWYU cleanup
#include "common.h"
#include "smart_holder_poc.h"
#include "typeid.h"

#include <memory>
#include <string>
#include <typeinfo>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)
PYBIND11_NAMESPACE_BEGIN(smart_holder_value_and_holder_support)

// BAKEIN_WIP: Factor out `struct value_and_holder` from type_caster_base.h,
//             then this helper does not have to be templated.
template <typename VHType>
struct value_and_holder_helper {
    const VHType *loaded_v_h;

    explicit value_and_holder_helper(const VHType *loaded_v_h) : loaded_v_h{loaded_v_h} {}

    bool have_holder() const {
        return loaded_v_h->vh != nullptr && loaded_v_h->holder_constructed();
    }

    pybindit::memory::smart_holder &holder() const {
        return loaded_v_h->template holder<pybindit::memory::smart_holder>();
    }

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
};

PYBIND11_NAMESPACE_END(smart_holder_value_and_holder_support)
PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
