/*
    pybind11/stdlib_enum.h: Declaration and conversion enums as Enum objects.

    Copyright (c) 2020 Ashley Whetter <ashley@awhetter.co.uk>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "detail/common.h"
#include "pybind11.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

template<typename U>
struct enum_mapper {
    handle type = {};
    std::unordered_map<U, handle> values = {};

    enum_mapper(handle type, const dict& values) : type(type) {
        for (auto item : values) {
            this->values[item.second.cast<U>()] = type.attr(item.first);
        }
    }
};

template<typename T>
struct type_caster<T, enable_if_t<std::is_enum<T>::value>> {
    using underlying_type = typename std::underlying_type<T>::type;

    private:
    using base_caster = type_caster_base<T>;
    using shared_info_type = typename std::unordered_map<std::type_index, void*>;

    static enum_mapper<underlying_type>* enum_info() {
        auto shared_enum_info = reinterpret_cast<shared_info_type*>(
            get_shared_data("_stdlib_enum_internals")
        );
        if (shared_enum_info) {
            auto it = shared_enum_info->find(std::type_index(typeid(T)));
            if (it != shared_enum_info->end()) {
                return reinterpret_cast<enum_mapper<underlying_type>*>(it->second);
            }
        }

        return nullptr;
    }

    base_caster caster;
    T value;

    public:
    template<typename U> using cast_op_type = pybind11::detail::cast_op_type<U>;

    operator T*() { return enum_info() ? &value: static_cast<T*>(caster); }
    operator T&() { return enum_info() ? value: static_cast<T&>(caster); }

    static constexpr auto name = base_caster::name;

    static handle cast(const T& src, return_value_policy policy, handle parent) {
        enum_mapper<underlying_type>* info = enum_info();
        if (info) {
            auto it = info->values.find(static_cast<underlying_type>(src));
            if (it != info->values.end()) {
                return it->second.inc_ref();
            }
        }

        return base_caster::cast(src, policy, parent);
    }

    bool load(handle src, bool convert) {
        if (!src) {
            return false;
        }

        enum_mapper<underlying_type>* info = enum_info();
        if (info) {
            if (!isinstance(src, info->type)) {
                return false;
            }

            value = static_cast<T>(src.attr("value").cast<underlying_type>());
            return true;
        }

        return caster.load(src, convert);
    }

    static void bind(handle type, const dict& values) {
        enum_mapper<underlying_type>* info = enum_info();
        delete info;

        auto shared_enum_info = &get_or_create_shared_data<shared_info_type>("_stdlib_enum_internals");
        (*shared_enum_info)[std::type_index(typeid(T))] = reinterpret_cast<void*>(
            new enum_mapper<underlying_type>(type, values)
        );
        set_shared_data("_stdlib_enum_internals", shared_enum_info);
    }
};

PYBIND11_NAMESPACE_END(detail)

template<typename T>
class stdlib_enum {
    public:
    using underlying_type = typename std::underlying_type<T>::type;

    stdlib_enum(handle scope, const char* name)
        : scope(scope), name(name)
    {
        kwargs["value"] = cast(name);
        kwargs["names"] = entries;
        if (scope) {
            if (hasattr(scope, "__module__")) {
                kwargs["module"] = scope.attr("__module__");
            }
            else if (hasattr(scope, "__name__")) {
                kwargs["module"] = scope.attr("__name__");
            }
#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 3
            if (hasattr(scope, "__qualname__")) {
                kwargs["qualname"] = scope.attr("__qualname__").cast<std::string>() + "." + name;
            }
#endif
        }
    }

    ~stdlib_enum() {
        object ctor = module::import("enum").attr("Enum");
        object unique = module::import("enum").attr("unique");
        object type = unique(ctor(**kwargs));
        setattr(scope, name, type);
        detail::type_caster<T>::bind(type, entries);
    }

    stdlib_enum& value(const char* name, T value) & {
        add_entry(name, value);
        return *this;
    }

    stdlib_enum&& value(const char* name, T value) && {
        add_entry(name, value);
        return std::move(*this);
    }

    private:
    handle scope;
    const char* name;
    dict entries;
    dict kwargs;

    void add_entry(const char* name, T value) {
        entries[name] = cast(static_cast<underlying_type>(value));
    }
};

PYBIND11_NAMESPACE_END(pybind11)
