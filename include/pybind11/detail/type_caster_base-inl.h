/*
    pybind11/detail/type_caster_base-inl.h: Out-of-line definitions for type_caster_base.h

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

// Every function defined here must start with PYBIND11_INLINE (or
// PYBIND11_NOINLINE_ATTR PYBIND11_INLINE). In the default header-only mode this file is
// included at the bottom of type_caster_base.h; when PYBIND11_PRECOMPILED is defined it
// is only compiled into the pybind11 static library (see src/).

#pragma once

#include "type_caster_base.h"

#include <string>
#include <utility>
#include <vector>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

PYBIND11_INLINE loader_life_support *&loader_life_support::tls_current_frame() {
    static thread_local loader_life_support *frame_ptr = nullptr;
    return frame_ptr;
}

PYBIND11_INLINE loader_life_support::loader_life_support() {
    auto &frame = tls_current_frame();
    parent = frame;
    frame = this;
}

PYBIND11_INLINE loader_life_support::~loader_life_support() {
    auto &frame = tls_current_frame();
    if (frame != this) {
        pybind11_fail("loader_life_support: internal error");
    }
    frame = parent;
    for (auto *item : keep_alive) {
        Py_DECREF(item);
    }
}

PYBIND11_NOINLINE_ATTR PYBIND11_INLINE bool loader_life_support::try_add_patient(handle h) {
    loader_life_support *frame = tls_current_frame();
    if (!frame) {
        return false;
    }
    if (frame->keep_alive.insert(h.ptr()).second) {
        Py_INCREF(h.ptr());
    }
    return true;
}

PYBIND11_NOINLINE_ATTR PYBIND11_INLINE void loader_life_support::add_patient(handle h) {
    if (!try_add_patient(h)) {
        // NOTE: It would be nice to include the stack frames here, as this indicates
        // use of pybind11::cast<> outside the normal call framework, finding such
        // a location is challenging. Developers could consider printing out
        // stack frame addresses here using something like __builtin_frame_address(0)
        throw cast_error("When called outside a bound function, py::cast() cannot "
                         "do Python -> C++ conversions which require the creation "
                         "of temporary values");
    }
}

// Band-aid workaround to fix a subtle but serious bug in a minimalistic fashion. See PR #4762.
PYBIND11_INLINE void all_type_info_add_base_most_derived_first(std::vector<type_info *> &bases,
                                                      type_info *addl_base) {
    for (auto it = bases.begin(); it != bases.end(); it++) {
        type_info *existing_base = *it;
        if (PyType_IsSubtype(addl_base->type, existing_base->type) != 0) {
            bases.insert(it, addl_base);
            return;
        }
    }
    bases.push_back(addl_base);
}

PYBIND11_NOINLINE_ATTR PYBIND11_INLINE void all_type_info_populate(PyTypeObject *t, std::vector<type_info *> &bases) {
    assert(bases.empty());
    std::vector<PyTypeObject *> check;
    for (handle parent : reinterpret_borrow<tuple>(t->tp_bases)) {
        check.push_back(reinterpret_cast<PyTypeObject *>(parent.ptr()));
    }
    auto const &type_dict = get_internals().registered_types_py;
    for (size_t i = 0; i < check.size(); i++) {
        auto *type = check[i];
        // Ignore Python2 old-style class super type:
        if (!PyType_Check((PyObject *) type)) {
            continue;
        }

        // Check `type` in the current set of registered python types:
        auto it = type_dict.find(type);
        if (it != type_dict.end()) {
            // We found a cache entry for it, so it's either pybind-registered or has pre-computed
            // pybind bases, but we have to make sure we haven't already seen the type(s) before:
            // we want to follow Python/virtual C++ rules that there should only be one instance of
            // a common base.
            for (auto *tinfo : it->second) {
                // NB: Could use a second set here, rather than doing a linear search, but since
                // having a large number of immediate pybind11-registered types seems fairly
                // unlikely, that probably isn't worthwhile.
                bool found = false;
                for (auto *known : bases) {
                    if (known == tinfo) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    all_type_info_add_base_most_derived_first(bases, tinfo);
                }
            }
        } else if (type->tp_bases) {
            // It's some python type, so keep follow its bases classes to look for one or more
            // registered types
            if (i + 1 == check.size()) {
                // When we're at the end, we can pop off the current element to avoid growing
                // `check` when adding just one base (which is typical--i.e. when there is no
                // multiple inheritance)
                check.pop_back();
                i--;
            }
            for (handle parent : reinterpret_borrow<tuple>(type->tp_bases)) {
                check.push_back(reinterpret_cast<PyTypeObject *>(parent.ptr()));
            }
        }
    }
}

PYBIND11_INLINE const std::vector<detail::type_info *> &all_type_info(PyTypeObject *type) {
    return all_type_info_get_cache(type).first->second;
}

PYBIND11_NOINLINE_ATTR PYBIND11_INLINE detail::type_info *get_type_info(PyTypeObject *type) {
    const auto &bases = all_type_info(type);
    if (bases.empty()) {
        return nullptr;
    }
    if (bases.size() > 1) {
        pybind11_fail(
            "pybind11::detail::get_type_info: type has multiple pybind11-registered bases");
    }
    return bases.front();
}

PYBIND11_INLINE detail::type_info *get_local_type_info_lock_held(const std::type_info &tp) {
    const auto &locals = get_local_internals().registered_types_cpp;
    auto it = locals.find(&tp);
    if (it != locals.end()) {
        return it->second;
    }
    return nullptr;
}

PYBIND11_INLINE detail::type_info *get_local_type_info(const std::type_info &tp) {
    // NB: internals and local_internals share a single mutex
    PYBIND11_LOCK_INTERNALS(get_internals());
    return get_local_type_info_lock_held(tp);
}

PYBIND11_INLINE detail::type_info *get_global_type_info_lock_held(const std::type_info &tp) {
    // This is a two-level lookup. Hopefully we find the type info in
    // registered_types_cpp_fast, but if not we try
    // registered_types_cpp and fill registered_types_cpp_fast for
    // next time.
    detail::type_info *type_info = nullptr;
    auto &internals = get_internals();
#if PYBIND11_INTERNALS_VERSION >= 12
    auto &fast_types = internals.registered_types_cpp_fast;
#endif
    auto &types = internals.registered_types_cpp;
#if PYBIND11_INTERNALS_VERSION >= 12
    auto fast_it = fast_types.find(&tp);
    if (fast_it != fast_types.end()) {
#    ifndef NDEBUG
        auto types_it = types.find(std::type_index(tp));
        assert(types_it != types.end());
        assert(types_it->second == fast_it->second);
#    endif
        return fast_it->second;
    }
#endif // PYBIND11_INTERNALS_VERSION >= 12

    auto it = types.find(std::type_index(tp));
    if (it != types.end()) {
#if PYBIND11_INTERNALS_VERSION >= 12
        // We found the type in the slow map but not the fast one, so
        // some other DSO added it (otherwise it would be in the fast
        // map under &tp) and therefore we must be an alias. Record
        // that.
        it->second->alias_chain.push_front(&tp);
        fast_types.emplace(&tp, it->second);
#endif
        type_info = it->second;
    }
    return type_info;
}

PYBIND11_INLINE detail::type_info *get_global_type_info(const std::type_info &tp) {
    PYBIND11_LOCK_INTERNALS(get_internals());
    return get_global_type_info_lock_held(tp);
}

PYBIND11_NOINLINE_ATTR PYBIND11_INLINE detail::type_info *get_type_info(const std::type_info &tp,
                                                   bool throw_if_missing) {
    PYBIND11_LOCK_INTERNALS(get_internals());
    if (auto *ltype = get_local_type_info_lock_held(tp)) {
        return ltype;
    }
    if (auto *gtype = get_global_type_info_lock_held(tp)) {
        return gtype;
    }

    if (throw_if_missing) {
        std::string tname = tp.name();
        detail::clean_type_id(tname);
        pybind11_fail("pybind11::detail::get_type_info: unable to find type info for \""
                      + std::move(tname) + '"');
    }
    return nullptr;
}

PYBIND11_NOINLINE_ATTR PYBIND11_INLINE handle get_type_handle(const std::type_info &tp, bool throw_if_missing) {
    detail::type_info *type_info = get_type_info(tp, throw_if_missing);
    return handle(type_info ? (reinterpret_cast<PyObject *>(type_info->type)) : nullptr);
}

PYBIND11_INLINE bool try_incref(PyObject *obj) {
    // Tries to increment the reference count of an object if it's not zero.
#if defined(Py_GIL_DISABLED) && PY_VERSION_HEX >= 0x030E00A4
    return PyUnstable_TryIncRef(obj);
#elif defined(Py_GIL_DISABLED)
    // See
    // https://github.com/python/cpython/blob/d05140f9f77d7dfc753dd1e5ac3a5962aaa03eff/Include/internal/pycore_object.h#L761
    uint32_t local = _Py_atomic_load_uint32_relaxed(&obj->ob_ref_local);
    local += 1;
    if (local == 0) {
        // immortal
        return true;
    }
    if (_Py_IsOwnedByCurrentThread(obj)) {
        _Py_atomic_store_uint32_relaxed(&obj->ob_ref_local, local);
#    ifdef Py_REF_DEBUG
        _Py_INCREF_IncRefTotal();
#    endif
        return true;
    }
    Py_ssize_t shared = _Py_atomic_load_ssize_relaxed(&obj->ob_ref_shared);
    for (;;) {
        // If the shared refcount is zero and the object is either merged
        // or may not have weak references, then we cannot incref it.
        if (shared == 0 || shared == _Py_REF_MERGED) {
            return false;
        }

        if (_Py_atomic_compare_exchange_ssize(
                &obj->ob_ref_shared, &shared, shared + (1 << _Py_REF_SHARED_SHIFT))) {
#    ifdef Py_REF_DEBUG
            _Py_INCREF_IncRefTotal();
#    endif
            return true;
        }
    }
#else
    assert(Py_REFCNT(obj) > 0);
    Py_INCREF(obj);
    return true;
#endif
}

PYBIND11_NOINLINE_ATTR PYBIND11_INLINE handle find_registered_python_instance(void *src,
                                                         const detail::type_info *tinfo) {
    return with_instance_map(src, [&](instance_map &instances) {
        auto it_instances = instances.equal_range(src);
        for (auto it_i = it_instances.first; it_i != it_instances.second; ++it_i) {
            for (auto *instance_type : detail::all_type_info(Py_TYPE(it_i->second))) {
                if (instance_type && same_type(*instance_type->cpptype, *tinfo->cpptype)) {
                    auto *wrapper = reinterpret_cast<PyObject *>(it_i->second);
                    if (try_incref(wrapper)) {
                        return handle(wrapper);
                    }
                }
            }
        }
        return handle();
    });
}

PYBIND11_NOINLINE_ATTR PYBIND11_INLINE value_and_holder
instance::get_value_and_holder(const type_info *find_type /*= nullptr default in common.h*/,
                               bool throw_if_missing /*= true in common.h*/) {
    // Optimize common case:
    if (!find_type || Py_TYPE(this) == find_type->type) {
        return value_and_holder(this, find_type, 0, 0);
    }

    detail::values_and_holders vhs(this);
    auto it = vhs.find(find_type);
    if (it != vhs.end()) {
        return *it;
    }

    if (!throw_if_missing) {
        return value_and_holder();
    }

#if defined(PYBIND11_DETAILED_ERROR_MESSAGES)
    pybind11_fail("pybind11::detail::instance::get_value_and_holder: `"
                  + get_fully_qualified_tp_name(find_type->type)
                  + "' is not a pybind11 base of the given `"
                  + get_fully_qualified_tp_name(Py_TYPE(this)) + "' instance");
#else
    pybind11_fail(
        "pybind11::detail::instance::get_value_and_holder: "
        "type is not a pybind11 base of the given instance "
        "(#define PYBIND11_DETAILED_ERROR_MESSAGES or compile in debug mode for type details)");
#endif
}

PYBIND11_NOINLINE_ATTR PYBIND11_INLINE void instance::allocate_layout() {
    const auto &tinfo = all_type_info(Py_TYPE(this));

    const size_t n_types = tinfo.size();

    if (n_types == 0) {
        pybind11_fail(
            "instance allocation failed: new instance has no pybind11-registered base types");
    }

    simple_layout
        = n_types == 1 && tinfo.front()->holder_size_in_ptrs <= instance_simple_holder_in_ptrs();

    // Simple path: no python-side multiple inheritance, and a small-enough holder
    if (simple_layout) {
        simple_value_holder[0] = nullptr;
        simple_holder_constructed = false;
        simple_instance_registered = false;
    } else { // multiple base types or a too-large holder
        // Allocate space to hold: [v1*][h1][v2*][h2]...[bb...] where [vN*] is a value pointer,
        // [hN] is the (uninitialized) holder instance for value N, and [bb...] is a set of bool
        // values that tracks whether each associated holder has been initialized.  Each [block] is
        // padded, if necessary, to an integer multiple of sizeof(void *).
        size_t space = 0;
        for (auto *t : tinfo) {
            space += 1;                      // value pointer
            space += t->holder_size_in_ptrs; // holder instance
        }
        size_t flags_at = space;
        space += size_in_ptrs(n_types); // status bytes (holder_constructed and
                                        // instance_registered)

        // Allocate space for flags, values, and holders, and initialize it to 0 (flags and values,
        // in particular, need to be 0).  Use Python's memory allocation
        // functions: Python is using pymalloc, which is designed to be
        // efficient for small allocations like the one we're doing here;
        // for larger allocations they are just wrappers around malloc.
        // TODO: is this still true for pure Python 3.6?
        nonsimple.values_and_holders = static_cast<void **>(PyMem_Calloc(space, sizeof(void *)));
        if (!nonsimple.values_and_holders) {
            throw std::bad_alloc();
        }
        nonsimple.status
            = reinterpret_cast<std::uint8_t *>(&nonsimple.values_and_holders[flags_at]);
    }
    owned = true;
}

// NOLINTNEXTLINE(readability-make-member-function-const)
PYBIND11_NOINLINE_ATTR PYBIND11_INLINE void instance::deallocate_layout() {
    if (!simple_layout) {
        PyMem_Free(reinterpret_cast<void *>(nonsimple.values_and_holders));
    }
}

PYBIND11_NOINLINE_ATTR PYBIND11_INLINE bool isinstance_generic(handle obj, const std::type_info &tp) {
    handle type = detail::get_type_handle(tp, false);
    if (!type) {
        return false;
    }
    return isinstance(obj, type);
}

PYBIND11_NOINLINE_ATTR PYBIND11_INLINE handle get_object_handle(const void *ptr, const detail::type_info *type) {
    return with_instance_map(ptr, [&](instance_map &instances) {
        auto range = instances.equal_range(ptr);
        for (auto it = range.first; it != range.second; ++it) {
            for (const auto &vh : values_and_holders(it->second)) {
                if (vh.type == type) {
                    return handle(reinterpret_cast<PyObject *>(it->second));
                }
            }
        }
        return handle();
    });
}

PYBIND11_INLINE object cpp_conduit_method(handle self,
                                 const bytes &pybind11_platform_abi_id,
                                 const capsule &cpp_type_info_capsule,
                                 const bytes &pointer_kind) {
#ifdef PYBIND11_HAS_STRING_VIEW
    using cpp_str = std::string_view;
#else
    using cpp_str = std::string;
#endif
    if (cpp_str(pybind11_platform_abi_id) != PYBIND11_PLATFORM_ABI_ID) {
        return none();
    }
    if (std::strcmp(cpp_type_info_capsule.name(), typeid(std::type_info).name()) != 0) {
        return none();
    }
    if (cpp_str(pointer_kind) != "raw_pointer_ephemeral") {
        throw std::runtime_error("Invalid pointer_kind: \"" + std::string(pointer_kind) + "\"");
    }
    const auto *cpp_type_info = cpp_type_info_capsule.get_pointer<const std::type_info>();
    type_caster_generic caster(*cpp_type_info);
    if (!caster.load(self, false)) {
        return none();
    }
    return capsule(caster.value, cpp_type_info->name());
}

PYBIND11_INLINE std::string quote_cpp_type_name(const std::string &cpp_type_name) {
    return cpp_type_name; // No-op for now. See PR #4888
}

PYBIND11_NOINLINE_ATTR PYBIND11_INLINE std::string type_info_description(const std::type_info &ti) {
    if (auto *type_data = get_type_info(ti)) {
        handle th(reinterpret_cast<PyObject *>(type_data->type));
        return th.attr("__module__").cast<std::string>() + '.'
               + th.attr("__qualname__").cast<std::string>();
    }
    return quote_cpp_type_name(clean_type_id(ti.name()));
}

PYBIND11_NOINLINE_ATTR PYBIND11_INLINE
type_caster_generic::type_caster_generic(const std::type_info &type_info)
    : typeinfo(get_type_info(type_info)), cpptype(&type_info) {}

PYBIND11_NOINLINE_ATTR PYBIND11_INLINE handle
type_caster_generic::cast(const cast_sources &srcs,
                          return_value_policy policy,
                          handle parent,
                          void *(*copy_constructor)(const void *),
                          void *(*move_constructor)(const void *),
                          const void *existing_holder) {
    if (!srcs.result.tinfo) {
        // No pybind11 type info. Raise an exception.
        std::string tname = srcs.downcast.cpptype   ? srcs.downcast.cpptype->name()
                            : srcs.original.cpptype ? srcs.original.cpptype->name()
                                                    : "<unspecified>";
        detail::clean_type_id(tname);
        std::string msg = "Unregistered type : " + tname;
        set_error(PyExc_TypeError, msg.c_str());
        return handle();
    }

    void *src = const_cast<void *>(srcs.result.cppobj);
    if (src == nullptr) {
        return none().release();
    }
    const type_info *tinfo = srcs.result.tinfo;

    if (handle registered_inst = find_registered_python_instance(src, tinfo)) {
        return registered_inst;
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
#if defined(PYBIND11_DETAILED_ERROR_MESSAGES)
                std::string type_name(tinfo->cpptype->name());
                detail::clean_type_id(type_name);
                throw cast_error("return_value_policy = copy, but type " + type_name
                                 + " is non-copyable!");
#else
                throw cast_error("return_value_policy = copy, but type is "
                                 "non-copyable! (#define PYBIND11_DETAILED_ERROR_MESSAGES or "
                                 "compile in debug mode for details)");
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
#if defined(PYBIND11_DETAILED_ERROR_MESSAGES)
                std::string type_name(tinfo->cpptype->name());
                detail::clean_type_id(type_name);
                throw cast_error("return_value_policy = move, but type " + type_name
                                 + " is neither movable nor copyable!");
#else
                throw cast_error("return_value_policy = move, but type is neither "
                                 "movable nor copyable! "
                                 "(#define PYBIND11_DETAILED_ERROR_MESSAGES or compile in "
                                 "debug mode for details)");
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

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
