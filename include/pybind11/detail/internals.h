/*
    pybind11/detail/internals.h: Internal data structure and related functions

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include <pybind11/conduit/pybind11_platform_abi_id.h>
#include <pybind11/gil_simple.h>
#include <pybind11/pytypes.h>

#include "common.h"

#include <exception>
#include <mutex>
#include <thread>

/// Tracks the `internals` and `type_info` ABI version independent of the main library version.
///
/// Some portions of the code use an ABI that is conditional depending on this
/// version number.  That allows ABI-breaking changes to be "pre-implemented".
/// Once the default version number is incremented, the conditional logic that
/// no longer applies can be removed.  Additionally, users that need not
/// maintain ABI compatibility can increase the version number in order to take
/// advantage of any functionality/efficiency improvements that depend on the
/// newer ABI.
///
/// WARNING: If you choose to manually increase the ABI version, note that
/// pybind11 may not be tested as thoroughly with a non-default ABI version, and
/// further ABI-incompatible changes may be made before the ABI is officially
/// changed to the new version.
#ifndef PYBIND11_INTERNALS_VERSION
#    define PYBIND11_INTERNALS_VERSION 9
#endif

#if PYBIND11_INTERNALS_VERSION < 9
#    error "PYBIND11_INTERNALS_VERSION 9 is the minimum for all platforms for pybind11v3."
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

using ExceptionTranslator = void (*)(std::exception_ptr);

PYBIND11_NAMESPACE_BEGIN(detail)

constexpr const char *internals_function_record_capsule_name = "pybind11_function_record_capsule";

// Forward declarations
inline PyTypeObject *make_static_property_type();
inline PyTypeObject *make_default_metaclass();
inline PyObject *make_object_base_type(PyTypeObject *metaclass);

// The old Python Thread Local Storage (TLS) API is deprecated in Python 3.7 in favor of the new
// Thread Specific Storage (TSS) API.
// Avoid unnecessary allocation of `Py_tss_t`, since we cannot use
// `Py_LIMITED_API` anyway.
#define PYBIND11_TLS_KEY_REF Py_tss_t &
#if defined(__clang__)
#    define PYBIND11_TLS_KEY_INIT(var)                                                            \
        _Pragma("clang diagnostic push")                                         /**/             \
            _Pragma("clang diagnostic ignored \"-Wmissing-field-initializers\"") /**/             \
            Py_tss_t var                                                                          \
            = Py_tss_NEEDS_INIT;                                                                  \
        _Pragma("clang diagnostic pop")
#elif defined(__GNUC__) && !defined(__INTEL_COMPILER)
#    define PYBIND11_TLS_KEY_INIT(var)                                                            \
        _Pragma("GCC diagnostic push")                                         /**/               \
            _Pragma("GCC diagnostic ignored \"-Wmissing-field-initializers\"") /**/               \
            Py_tss_t var                                                                          \
            = Py_tss_NEEDS_INIT;                                                                  \
        _Pragma("GCC diagnostic pop")
#else
#    define PYBIND11_TLS_KEY_INIT(var) Py_tss_t var = Py_tss_NEEDS_INIT;
#endif
#define PYBIND11_TLS_KEY_CREATE(var) (PyThread_tss_create(&(var)) == 0)
#define PYBIND11_TLS_GET_VALUE(key) PyThread_tss_get(&(key))
#define PYBIND11_TLS_REPLACE_VALUE(key, value) PyThread_tss_set(&(key), (value))
#define PYBIND11_TLS_DELETE_VALUE(key) PyThread_tss_set(&(key), nullptr)
#define PYBIND11_TLS_FREE(key) PyThread_tss_delete(&(key))

// Python loads modules by default with dlopen with the RTLD_LOCAL flag; under libc++ and possibly
// other STLs, this means `typeid(A)` from one module won't equal `typeid(A)` from another module
// even when `A` is the same, non-hidden-visibility type (e.g. from a common include).  Under
// libstdc++, this doesn't happen: equality and the type_index hash are based on the type name,
// which works.  If not under a known-good stl, provide our own name-based hash and equality
// functions that use the type name.
#if !defined(_LIBCPP_VERSION)
inline bool same_type(const std::type_info &lhs, const std::type_info &rhs) { return lhs == rhs; }
using type_hash = std::hash<std::type_index>;
using type_equal_to = std::equal_to<std::type_index>;
#else
inline bool same_type(const std::type_info &lhs, const std::type_info &rhs) {
    return lhs.name() == rhs.name() || std::strcmp(lhs.name(), rhs.name()) == 0;
}

struct type_hash {
    size_t operator()(const std::type_index &t) const {
        size_t hash = 5381;
        const char *ptr = t.name();
        while (auto c = static_cast<unsigned char>(*ptr++)) {
            hash = (hash * 33) ^ c;
        }
        return hash;
    }
};

struct type_equal_to {
    bool operator()(const std::type_index &lhs, const std::type_index &rhs) const {
        return lhs.name() == rhs.name() || std::strcmp(lhs.name(), rhs.name()) == 0;
    }
};
#endif

template <typename value_type>
using type_map = std::unordered_map<std::type_index, value_type, type_hash, type_equal_to>;

struct override_hash {
    inline size_t operator()(const std::pair<const PyObject *, const char *> &v) const {
        size_t value = std::hash<const void *>()(v.first);
        value ^= std::hash<const void *>()(v.second) + 0x9e3779b9 + (value << 6) + (value >> 2);
        return value;
    }
};

using instance_map = std::unordered_multimap<const void *, instance *>;

#ifdef Py_GIL_DISABLED
// Wrapper around PyMutex to provide BasicLockable semantics
class pymutex {
    PyMutex mutex;

public:
    pymutex() : mutex({}) {}
    void lock() { PyMutex_Lock(&mutex); }
    void unlock() { PyMutex_Unlock(&mutex); }
};

// Instance map shards are used to reduce mutex contention in free-threaded Python.
struct instance_map_shard {
    instance_map registered_instances;
    pymutex mutex;
    // alignas(64) would be better, but causes compile errors in macOS before 10.14 (see #5200)
    char padding[64 - (sizeof(instance_map) + sizeof(pymutex)) % 64];
};

static_assert(sizeof(instance_map_shard) % 64 == 0,
              "instance_map_shard size is not a multiple of 64 bytes");
#endif

/// Internal data structure used to track registered instances and types.
/// Whenever binary incompatible changes are made to this structure,
/// `PYBIND11_INTERNALS_VERSION` must be incremented.
struct internals {
#ifdef Py_GIL_DISABLED
    pymutex mutex;
    pymutex exception_translator_mutex;
#endif
    // std::type_index -> pybind11's type information
    type_map<type_info *> registered_types_cpp;
    // PyTypeObject* -> base type_info(s)
    std::unordered_map<PyTypeObject *, std::vector<type_info *>> registered_types_py;
#ifdef Py_GIL_DISABLED
    std::unique_ptr<instance_map_shard[]> instance_shards; // void * -> instance*
    size_t instance_shards_mask;
#else
    instance_map registered_instances; // void * -> instance*
#endif
    std::unordered_set<std::pair<const PyObject *, const char *>, override_hash>
        inactive_override_cache;
    type_map<std::vector<bool (*)(PyObject *, void *&)>> direct_conversions;
    std::unordered_map<const PyObject *, std::vector<PyObject *>> patients;
    std::forward_list<ExceptionTranslator> registered_exception_translators;
    std::unordered_map<std::string, void *> shared_data; // Custom data to be shared across
                                                         // extensions
    std::forward_list<std::string> static_strings;       // Stores the std::strings backing
                                                         // detail::c_str()
    PyTypeObject *static_property_type;
    PyTypeObject *default_metaclass;
    PyObject *instance_base;
    // Unused if PYBIND11_SIMPLE_GIL_MANAGEMENT is defined:
    PYBIND11_TLS_KEY_INIT(tstate)
    PYBIND11_TLS_KEY_INIT(loader_life_support_tls_key)
    // Unused if PYBIND11_SIMPLE_GIL_MANAGEMENT is defined:
    PyInterpreterState *istate = nullptr;

    type_map<PyObject *> native_enum_type_map;

    internals() = default;
    internals(const internals &other) = delete;
    internals &operator=(const internals &other) = delete;
    ~internals() {
        PYBIND11_TLS_FREE(loader_life_support_tls_key);

        // This destructor is called *after* Py_Finalize() in finalize_interpreter().
        // That *SHOULD BE* fine. The following details what happens when PyThread_tss_free is
        // called. PYBIND11_TLS_FREE is PyThread_tss_free on python 3.7+. On older python, it does
        // nothing. PyThread_tss_free calls PyThread_tss_delete and PyMem_RawFree.
        // PyThread_tss_delete just calls TlsFree (on Windows) or pthread_key_delete (on *NIX).
        // Neither of those have anything to do with CPython internals. PyMem_RawFree *requires*
        // that the `tstate` be allocated with the CPython allocator.
        PYBIND11_TLS_FREE(tstate);
    }
};

// For backwards compatibility (i.e. #ifdef guards):
#define PYBIND11_HAS_INTERNALS_WITH_SMART_HOLDER_SUPPORT

enum class holder_enum_t : uint8_t {
    undefined,
    std_unique_ptr, // Default, lacking interop with std::shared_ptr.
    std_shared_ptr, // Lacking interop with std::unique_ptr.
    smart_holder,   // Full std::unique_ptr / std::shared_ptr interop.
    custom_holder,
};

/// Additional type information which does not fit into the PyTypeObject.
/// Changes to this struct also require bumping `PYBIND11_INTERNALS_VERSION`.
struct type_info {
    PyTypeObject *type;
    const std::type_info *cpptype;
    size_t type_size, type_align, holder_size_in_ptrs;
    void *(*operator_new)(size_t);
    void (*init_instance)(instance *, const void *);
    void (*dealloc)(value_and_holder &v_h);
    std::vector<PyObject *(*) (PyObject *, PyTypeObject *)> implicit_conversions;
    std::vector<std::pair<const std::type_info *, void *(*) (void *)>> implicit_casts;
    std::vector<bool (*)(PyObject *, void *&)> *direct_conversions;
    buffer_info *(*get_buffer)(PyObject *, void *) = nullptr;
    void *get_buffer_data = nullptr;
    void *(*module_local_load)(PyObject *, const type_info *) = nullptr;
    holder_enum_t holder_enum_v = holder_enum_t::undefined;
    /* A simple type never occurs as a (direct or indirect) parent
     * of a class that makes use of multiple inheritance.
     * A type can be simple even if it has non-simple ancestors as long as it has no descendants.
     */
    bool simple_type : 1;
    /* True if there is no multiple inheritance in this type's inheritance tree */
    bool simple_ancestors : 1;
    /* true if this is a type registered with py::module_local */
    bool module_local : 1;
};

#define PYBIND11_INTERNALS_ID                                                                     \
    "__pybind11_internals_v" PYBIND11_TOSTRING(PYBIND11_INTERNALS_VERSION)                        \
        PYBIND11_COMPILER_TYPE_LEADING_UNDERSCORE PYBIND11_PLATFORM_ABI_ID "__"

#define PYBIND11_MODULE_LOCAL_ID                                                                  \
    "__pybind11_module_local_v" PYBIND11_TOSTRING(PYBIND11_INTERNALS_VERSION)                     \
        PYBIND11_COMPILER_TYPE_LEADING_UNDERSCORE PYBIND11_PLATFORM_ABI_ID "__"

/// Each module locally stores a pointer to the `internals` data. The data
/// itself is shared among modules with the same `PYBIND11_INTERNALS_ID`.
inline internals **&get_internals_pp() {
    static internals **internals_pp = nullptr;
    return internals_pp;
}

// forward decl
inline void translate_exception(std::exception_ptr);

template <class T,
          enable_if_t<std::is_same<std::nested_exception, remove_cvref_t<T>>::value, int> = 0>
bool handle_nested_exception(const T &exc, const std::exception_ptr &p) {
    std::exception_ptr nested = exc.nested_ptr();
    if (nested != nullptr && nested != p) {
        translate_exception(nested);
        return true;
    }
    return false;
}

template <class T,
          enable_if_t<!std::is_same<std::nested_exception, remove_cvref_t<T>>::value, int> = 0>
bool handle_nested_exception(const T &exc, const std::exception_ptr &p) {
    if (const auto *nep = dynamic_cast<const std::nested_exception *>(std::addressof(exc))) {
        return handle_nested_exception(*nep, p);
    }
    return false;
}

inline bool raise_err(PyObject *exc_type, const char *msg) {
    if (PyErr_Occurred()) {
        raise_from(exc_type, msg);
        return true;
    }
    set_error(exc_type, msg);
    return false;
}

inline void translate_exception(std::exception_ptr p) {
    if (!p) {
        return;
    }
    try {
        std::rethrow_exception(p);
    } catch (error_already_set &e) {
        handle_nested_exception(e, p);
        e.restore();
        return;
    } catch (const builtin_exception &e) {
        // Could not use template since it's an abstract class.
        if (const auto *nep = dynamic_cast<const std::nested_exception *>(std::addressof(e))) {
            handle_nested_exception(*nep, p);
        }
        e.set_error();
        return;
    } catch (const std::bad_alloc &e) {
        handle_nested_exception(e, p);
        raise_err(PyExc_MemoryError, e.what());
        return;
    } catch (const std::domain_error &e) {
        handle_nested_exception(e, p);
        raise_err(PyExc_ValueError, e.what());
        return;
    } catch (const std::invalid_argument &e) {
        handle_nested_exception(e, p);
        raise_err(PyExc_ValueError, e.what());
        return;
    } catch (const std::length_error &e) {
        handle_nested_exception(e, p);
        raise_err(PyExc_ValueError, e.what());
        return;
    } catch (const std::out_of_range &e) {
        handle_nested_exception(e, p);
        raise_err(PyExc_IndexError, e.what());
        return;
    } catch (const std::range_error &e) {
        handle_nested_exception(e, p);
        raise_err(PyExc_ValueError, e.what());
        return;
    } catch (const std::overflow_error &e) {
        handle_nested_exception(e, p);
        raise_err(PyExc_OverflowError, e.what());
        return;
    } catch (const std::exception &e) {
        handle_nested_exception(e, p);
        raise_err(PyExc_RuntimeError, e.what());
        return;
    } catch (const std::nested_exception &e) {
        handle_nested_exception(e, p);
        raise_err(PyExc_RuntimeError, "Caught an unknown nested exception!");
        return;
    } catch (...) {
        raise_err(PyExc_RuntimeError, "Caught an unknown exception!");
        return;
    }
}

#if !defined(__GLIBCXX__)
inline void translate_local_exception(std::exception_ptr p) {
    try {
        if (p) {
            std::rethrow_exception(p);
        }
    } catch (error_already_set &e) {
        e.restore();
        return;
    } catch (const builtin_exception &e) {
        e.set_error();
        return;
    }
}
#endif

inline object get_python_state_dict() {
    object state_dict;
#if defined(PYPY_VERSION) || defined(GRAALVM_PYTHON)
    state_dict = reinterpret_borrow<object>(PyEval_GetBuiltins());
#else
#    if PY_VERSION_HEX < 0x03090000
    PyInterpreterState *istate = _PyInterpreterState_Get();
#    else
    PyInterpreterState *istate = PyInterpreterState_Get();
#    endif
    if (istate) {
        state_dict = reinterpret_borrow<object>(PyInterpreterState_GetDict(istate));
    }
#endif
    if (!state_dict) {
        raise_from(PyExc_SystemError, "pybind11::detail::get_python_state_dict() FAILED");
        throw error_already_set();
    }
    return state_dict;
}

inline object get_internals_obj_from_state_dict(handle state_dict) {
    return reinterpret_steal<object>(
        dict_getitemstringref(state_dict.ptr(), PYBIND11_INTERNALS_ID));
}

inline internals **get_internals_pp_from_capsule(handle obj) {
    void *raw_ptr = PyCapsule_GetPointer(obj.ptr(), /*name=*/nullptr);
    if (raw_ptr == nullptr) {
        raise_from(PyExc_SystemError, "pybind11::detail::get_internals_pp_from_capsule() FAILED");
        throw error_already_set();
    }
    return static_cast<internals **>(raw_ptr);
}

inline uint64_t round_up_to_next_pow2(uint64_t x) {
    // Round-up to the next power of two.
    // See https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
    x--;
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    x |= (x >> 32);
    x++;
    return x;
}

/// Return a reference to the current `internals` data
PYBIND11_NOINLINE internals &get_internals() {
    auto **&internals_pp = get_internals_pp();
    if (internals_pp && *internals_pp) {
        return **internals_pp;
    }

    gil_scoped_acquire_simple gil;
    error_scope err_scope;

    dict state_dict = get_python_state_dict();
    if (object internals_obj = get_internals_obj_from_state_dict(state_dict)) {
        internals_pp = get_internals_pp_from_capsule(internals_obj);
    }
    if (internals_pp && *internals_pp) {
        // We loaded the internals through `state_dict`, which means that our `error_already_set`
        // and `builtin_exception` may be different local classes than the ones set up in the
        // initial exception translator, below, so add another for our local exception classes.
        //
        // libstdc++ doesn't require this (types there are identified only by name)
        // libc++ with CPython doesn't require this (types are explicitly exported)
        // libc++ with PyPy still need it, awaiting further investigation
#if !defined(__GLIBCXX__)
        (*internals_pp)->registered_exception_translators.push_front(&translate_local_exception);
#endif
    } else {
        if (!internals_pp) {
            internals_pp = new internals *();
        }
        auto *&internals_ptr = *internals_pp;
        internals_ptr = new internals();

        PyThreadState *tstate = PyThreadState_Get();
        // NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
        if (!PYBIND11_TLS_KEY_CREATE(internals_ptr->tstate)) {
            pybind11_fail("get_internals: could not successfully initialize the tstate TSS key!");
        }
        PYBIND11_TLS_REPLACE_VALUE(internals_ptr->tstate, tstate);

        // NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
        if (!PYBIND11_TLS_KEY_CREATE(internals_ptr->loader_life_support_tls_key)) {
            pybind11_fail("get_internals: could not successfully initialize the "
                          "loader_life_support TSS key!");
        }

        internals_ptr->istate = tstate->interp;
        state_dict[PYBIND11_INTERNALS_ID] = capsule(reinterpret_cast<void *>(internals_pp));
        internals_ptr->registered_exception_translators.push_front(&translate_exception);
        internals_ptr->static_property_type = make_static_property_type();
        internals_ptr->default_metaclass = make_default_metaclass();
        internals_ptr->instance_base = make_object_base_type(internals_ptr->default_metaclass);
#ifdef Py_GIL_DISABLED
        // Scale proportional to the number of cores. 2x is a heuristic to reduce contention.
        auto num_shards
            = static_cast<size_t>(round_up_to_next_pow2(2 * std::thread::hardware_concurrency()));
        if (num_shards == 0) {
            num_shards = 1;
        }
        internals_ptr->instance_shards.reset(new instance_map_shard[num_shards]);
        internals_ptr->instance_shards_mask = num_shards - 1;
#endif // Py_GIL_DISABLED
    }
    return **internals_pp;
}

// the internals struct (above) is shared between all the modules. local_internals are only
// for a single module. Any changes made to internals may require an update to
// PYBIND11_INTERNALS_VERSION, breaking backwards compatibility. local_internals is, by design,
// restricted to a single module. Whether a module has local internals or not should not
// impact any other modules, because the only things accessing the local internals is the
// module that contains them.
struct local_internals {
    type_map<type_info *> registered_types_cpp;
    std::forward_list<ExceptionTranslator> registered_exception_translators;
};

/// Works like `get_internals`, but for things which are locally registered.
inline local_internals &get_local_internals() {
    // Current static can be created in the interpreter finalization routine. If the later will be
    // destroyed in another static variable destructor, creation of this static there will cause
    // static deinitialization fiasco. In order to avoid it we avoid destruction of the
    // local_internals static. One can read more about the problem and current solution here:
    // https://google.github.io/styleguide/cppguide.html#Static_and_Global_Variables
    static auto *locals = new local_internals();
    return *locals;
}

#ifdef Py_GIL_DISABLED
#    define PYBIND11_LOCK_INTERNALS(internals) std::unique_lock<pymutex> lock((internals).mutex)
#else
#    define PYBIND11_LOCK_INTERNALS(internals)
#endif

template <typename F>
inline auto with_internals(const F &cb) -> decltype(cb(get_internals())) {
    auto &internals = get_internals();
    PYBIND11_LOCK_INTERNALS(internals);
    return cb(internals);
}

template <typename F>
inline auto with_exception_translators(const F &cb)
    -> decltype(cb(get_internals().registered_exception_translators,
                   get_local_internals().registered_exception_translators)) {
    auto &internals = get_internals();
#ifdef Py_GIL_DISABLED
    std::unique_lock<pymutex> lock((internals).exception_translator_mutex);
#endif
    auto &local_internals = get_local_internals();
    return cb(internals.registered_exception_translators,
              local_internals.registered_exception_translators);
}

inline std::uint64_t mix64(std::uint64_t z) {
    // David Stafford's variant 13 of the MurmurHash3 finalizer popularized
    // by the SplitMix PRNG.
    // https://zimbry.blogspot.com/2011/09/better-bit-mixing-improving-on.html
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

template <typename F>
inline auto with_instance_map(const void *ptr, const F &cb)
    -> decltype(cb(std::declval<instance_map &>())) {
    auto &internals = get_internals();

#ifdef Py_GIL_DISABLED
    // Hash address to compute shard, but ignore low bits. We'd like allocations
    // from the same thread/core to map to the same shard and allocations from
    // other threads/cores to map to other shards. Using the high bits is a good
    // heuristic because memory allocators often have a per-thread
    // arena/superblock/segment from which smaller allocations are served.
    auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    auto hash = mix64(static_cast<std::uint64_t>(addr >> 20));
    auto idx = static_cast<size_t>(hash & internals.instance_shards_mask);

    auto &shard = internals.instance_shards[idx];
    std::unique_lock<pymutex> lock(shard.mutex);
    return cb(shard.registered_instances);
#else
    (void) ptr;
    return cb(internals.registered_instances);
#endif
}

// Returns the number of registered instances for testing purposes.  The result may not be
// consistent if other threads are registering or unregistering instances concurrently.
inline size_t num_registered_instances() {
    auto &internals = get_internals();
#ifdef Py_GIL_DISABLED
    size_t count = 0;
    for (size_t i = 0; i <= internals.instance_shards_mask; ++i) {
        auto &shard = internals.instance_shards[i];
        std::unique_lock<pymutex> lock(shard.mutex);
        count += shard.registered_instances.size();
    }
    return count;
#else
    return internals.registered_instances.size();
#endif
}

/// Constructs a std::string with the given arguments, stores it in `internals`, and returns its
/// `c_str()`.  Such strings objects have a long storage duration -- the internal strings are only
/// cleared when the program exits or after interpreter shutdown (when embedding), and so are
/// suitable for c-style strings needed by Python internals (such as PyTypeObject's tp_name).
template <typename... Args>
const char *c_str(Args &&...args) {
    // GCC 4.8 doesn't like parameter unpack within lambda capture, so use
    // PYBIND11_LOCK_INTERNALS.
    auto &internals = get_internals();
    PYBIND11_LOCK_INTERNALS(internals);
    auto &strings = internals.static_strings;
    strings.emplace_front(std::forward<Args>(args)...);
    return strings.front().c_str();
}

PYBIND11_NAMESPACE_END(detail)

/// Returns a named pointer that is shared among all extension modules (using the same
/// pybind11 version) running in the current interpreter. Names starting with underscores
/// are reserved for internal usage. Returns `nullptr` if no matching entry was found.
PYBIND11_NOINLINE void *get_shared_data(const std::string &name) {
    return detail::with_internals([&](detail::internals &internals) {
        auto it = internals.shared_data.find(name);
        return it != internals.shared_data.end() ? it->second : nullptr;
    });
}

/// Set the shared data that can be later recovered by `get_shared_data()`.
PYBIND11_NOINLINE void *set_shared_data(const std::string &name, void *data) {
    return detail::with_internals([&](detail::internals &internals) {
        internals.shared_data[name] = data;
        return data;
    });
}

/// Returns a typed reference to a shared data entry (by using `get_shared_data()`) if
/// such entry exists. Otherwise, a new object of default-constructible type `T` is
/// added to the shared data under the given name and a reference to it is returned.
template <typename T>
T &get_or_create_shared_data(const std::string &name) {
    return *detail::with_internals([&](detail::internals &internals) {
        auto it = internals.shared_data.find(name);
        T *ptr = (T *) (it != internals.shared_data.end() ? it->second : nullptr);
        if (!ptr) {
            ptr = new T();
            internals.shared_data[name] = ptr;
        }
        return ptr;
    });
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
