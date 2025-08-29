/*
 * pymetabind.h: definitions for interoperability between different
 *               Python binding frameworks
 *
 * Copy this header file into the implementation of a framework that uses it.
 * This functionality is intended to be used by the framework itself,
 * rather than by users of the framework.
 *
 * This is version 0.1+dev of pymetabind. Changelog:
 *
 *      Unreleased: Fix typo in Py_GIL_DISABLED. Add pymb_framework::leak_safe.
 *                  Add casts from PyTypeObject* to PyObject* where needed.
 *
 *     Version 0.1: Initial draft. ABI may change without warning while we
 *      2025-08-16  prove out the concept. Please wait for a 1.0 release
 *                  before including this header in a published release of
 *                  any binding framework.
 *
 * Copyright (c) 2025 Hudson River Trading <opensource@hudson-trading.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stddef.h>
#include <assert.h>

#if !defined(PY_VERSION_HEX)
#  error You must include Python.h before this header
#endif

/*
 * There are two ways to use this header file. The default is header-only style,
 * where all functions are defined as `inline`. If you want to emit functions
 * as non-inline, perhaps so you can link against them from non-C/C++ code,
 * then do the following:
 * - In every compilation unit that includes this header, `#define PYMB_FUNC`
 *   first. (The `PYMB_FUNC` macro will be expanded in place of the "inline"
 *   keyword, so you can also use it to add any other declaration attributes
 *   required by your environment.)
 * - In all those compilation units except one, also `#define PYMB_DECLS_ONLY`
 *   before including this header. The definitions will be emitted in the
 *   compilation unit that doesn't request `PYMB_DECLS_ONLY`.
 */
#if !defined(PYMB_FUNC)
#define PYMB_FUNC inline
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/*
 * Approach used to cast a previously unknown C++ instance into a Python object.
 * The values of these enumerators match those for `nanobind::rv_policy` and
 * `pybind11::return_value_policy`.
 */
enum pymb_rv_policy {
    // (Values 0 and 1 correspond to `automatic` and `automatic_reference`,
    //  which should become one of the other policies before reaching us)

    // Create a Python object that owns a pointer to heap-allocated storage
    // and will destroy and deallocate it when the Python object is destroyed
    pymb_rv_policy_take_ownership = 2,

    // Create a Python object that owns a new C++ instance created via
    // copy construction from the given one
    pymb_rv_policy_copy = 3,

    // Create a Python object that owns a new C++ instance created via
    // move construction from the given one
    pymb_rv_policy_move = 4,

    // Create a Python object that wraps the given pointer to a C++ instance
    // but will not destroy or deallocate it
    pymb_rv_policy_reference = 5,

    // `reference`, plus arrange for the given `parent` python object to
    // live at least as long as the new object that wraps the pointer
    pymb_rv_policy_reference_internal = 6,

    // Don't create a new Python object; only try to look up an existing one
    // from the same framework
    pymb_rv_policy_none = 7
};

/*
 * The language to which a particular framework provides bindings. Each
 * language has its own semantics for how to interpret
 * `pymb_framework::abi_extra` and `pymb_binding::native_type`.
 */
enum pymb_abi_lang {
    // C. `pymb_framework::abi_extra` and `pymb_binding::native_type` are NULL.
    pymb_abi_lang_c = 1,

    // C++. `pymb_framework::abi_extra` is in the format used by
    // nanobind since 2.6.1 (NB_PLATFORM_ABI_TAG in nanobind/src/nb_abi.h)
    // and pybind11 since 2.11.2/2.12.1/2.13.6 (PYBIND11_PLATFORM_ABI_ID in
    // pybind11/include/pybind11/conduit/pybind11_platform_abi_id.h).
    // `pymb_binding::native_type` is a cast `const std::type_info*` pointer.
    pymb_abi_lang_cpp = 2,

    // extensions welcome!
};

/*
 * Simple linked list implementation. `pymb_list_node` should be the first
 * member of a structure so you can downcast it to the appropriate type.
 */
struct pymb_list_node {
    struct pymb_list_node *next;
    struct pymb_list_node *prev;
};

struct pymb_list {
    struct pymb_list_node head;
};

inline void pymb_list_init(struct pymb_list* list) {
    list->head.prev = list->head.next = &list->head;
}

inline void pymb_list_unlink(struct pymb_list_node* node) {
    if (node->next) {
        node->next->prev = node->prev;
        node->prev->next = node->next;
        node->next = node->prev = NULL;
    }
}

inline void pymb_list_append(struct pymb_list* list,
                             struct pymb_list_node* node) {
    pymb_list_unlink(node);
    struct pymb_list_node* tail = list->head.prev;
    tail->next = node;
    list->head.prev = node;
    node->prev = tail;
    node->next = &list->head;
}

#define PYMB_LIST_FOREACH(type, name, list)      \
    for (type name = (type) (list).head.next;    \
         name != (type) &(list).head;            \
         name = (type) name->hook.next)

/*
 * The registry holds information about all the interoperable binding
 * frameworks and individual type bindings that are loaded in a Python
 * interpreter process. It is protected by a mutex in free-threaded builds,
 * and by the GIL in regular builds.
 *
 * The only data structure we use is a C doubly-linked list, which offers a
 * lowest-common-denominator ABI and cheap addition and removal. It is expected
 * that individual binding frameworks will use their `add_foreign_binding` and
 * `remove_foreign_binding` callbacks to maintain references to these structures
 * in more-performant private data structures of their choosing.
 *
 * The pointer to the registry is stored in a Python capsule object with type
 * "pymetabind_registry", which is stored in the PyInterpreterState_GetDict()
 * under the string key "__pymetabind_registry__". Any ABI-incompatible changes
 * after v1.0 (which we hope to avoid!) will result in a new name for the
 * dictionary key. You can obtain a registry pointer using
 * `pymb_get_registry()`, defined below.
 */
struct pymb_registry {
    // Linked list of registered `pymb_framework` structures
    struct pymb_list frameworks;

    // Linked list of registered `pymb_binding` structures
    struct pymb_list bindings;

    // Reserved for future extensions; currently set to 0
    uint32_t reserved;

#if defined(Py_GIL_DISABLED)
    // Mutex guarding accesses to `frameworks` and `bindings`.
    // On non-free-threading builds, these are guarded by the Python GIL.
    PyMutex mutex;
#endif
};

#if defined(Py_GIL_DISABLED)
inline void pymb_lock_registry(struct pymb_registry* registry) {
    PyMutex_Lock(&registry->mutex);
}
inline void pymb_unlock_registry(struct pymb_registry* registry) {
    PyMutex_Unlock(&registry->mutex);
}
#else
inline void pymb_lock_registry(struct pymb_registry*) {}
inline void pymb_unlock_registry(struct pymb_registry*) {}
#endif

struct pymb_binding;

/*
 * Information about one framework that has registered itself with pymetabind.
 * "Framework" here refers to a set of bindings that are natively mutually
 * interoperable. So, different binding libraries would be different frameworks,
 * as would versions of the same library that use incompatible data structures
 * due to ABI changes or build flags.
 *
 * A framework that wishes to either export bindings (allow other frameworks
 * to perform to/from Python conversion for its types) or import bindings
 * (perform its own to/from Python conversion for other frameworks' types)
 * must start by creating and filling out a `pymb_framework` structure.
 * This can be allocated in any way that the framework prefers (e.g., on
 * the heap or in static storage). Once filled out, the framework structure
 * should be passed to `pymb_add_framework()`. It must then remain accessible
 * and unmodified (except as documented below) until the Python interpreter
 * is finalized. After finalization, such as in a `Py_AtExit` handler, if
 * all bindings have been removed already, you may optionally clean up by
 * calling `pymb_list_unlink(&framework->hook)` and then deallocating the
 * `pymb_framework` structure.
 *
 * All fields of this structure are set before it is made visible to other
 * threads and then never changed, so they don't need locking to access.
 * Some methods require locking or other synchronization to call; see their
 * individual documentation.
 */
struct pymb_framework {
    // Hook by which this structure is linked into the list of
    // `pymb_registry::frameworks`. May be modified as other frameworks are
    // added; protected by the `pymb_registry::mutex` in free-threaded builds.
    struct pymb_list_node hook;

    // Human-readable description of this framework, as a NUL-terminated string
    const char* name;

    // Does this framework guarantee that its `pymb_binding` structures remain
    // valid to use for the lifetime of the Python interpreter process once
    // they have been linked into the lists in `pymb_registry`? Setting this
    // to true reduces the number of atomic operations needed to work with
    // this framework's bindings in free-threaded builds.
    uint8_t bindings_usable_forever;

    // Does this framework reliably deallocate all of its type and function
    // objects by the time the Python interpreter is finalized, in the absence
    // of bugs in user code? If not, it might cause leaks of other frameworks'
    // types or functions, via attributes or default argument values for
    // this framework's leaked objects.
    uint8_t leak_safe;

    // Reserved for future extensions. Set to 0.
    uint8_t reserved[2];

    // The language to which this framework provides bindings: one of the
    // `pymb_abi_lang` enumerators.
    enum pymb_abi_lang abi_lang;

    // NUL-terminated string constant encoding additional information that must
    // match in order for two types with the same `abi_lang` to be usable from
    // each other's environments. See documentation of `abi_lang` enumerators
    // for language-specific guidance. This may be NULL if there are no
    // additional ABI details that are relevant for your language.
    //
    // This is only the platform details that affect things like the layout
    // of objects provided by the `abi_lang` (std::string, etc); Python build
    // details (free-threaded, stable ABI, etc) should not impact this string.
    // Details that are already guaranteed to match by virtue of being in the
    // same address space -- architecture, pointer size, OS -- also should not
    // impact this string.
    //
    // For efficiency, `pymb_add_framework()` will compare this against every
    // other registered framework's `abi_extra` tag, and re-point an incoming
    // framework's `abi_extra` field to refer to the matching `abi_extra` string
    // of an already-registered framework if one exists. This acts as a simple
    // form of interning to speed up checking that a given binding is usable.
    // Thus, to check whether another framework's ABI matches yours, you can
    // do a pointer comparison `me->abi_extra == them->abi_extra`.
    const char* abi_extra;

    // The function pointers below allow other frameworks to interact with
    // bindings provided by this framework. They are constant after construction
    // and, except for `translate_exception()`, must not throw C++ exceptions.
    // Unless otherwise documented, they must not be NULL.

    // Extract a C/C++/etc object from `pyobj`. The desired type is specified by
    // providing a `pymb_binding*` for some binding that belongs to this
    // framework. Return a pointer to the object, or NULL if no pointer of the
    // appropriate type could be extracted.
    //
    // If `convert` is nonzero, be more willing to perform implicit conversions
    // to make the cast succeed; the intent is that one could perform overload
    // resolution by doing a first pass with convert=false to find an exact
    // match, and then a second with convert=true to find an approximate match
    // if there's no exact match.
    //
    // If `keep_referenced` is not NULL, then `from_python` may make calls to
    // `keep_referenced` to request that some Python objects remain referenced
    // until the returned object is no longer needed. The `keep_referenced_ctx`
    // will be passed as the first argument to any such calls.
    // `keep_referenced` should incref its `obj` immediately and remember
    // that it should be decref'ed later, for no net change in refcount.
    // This is an abstraction around something like the cleanup_list in
    // nanobind or loader_life_support in pybind11.
    //
    // On free-threaded builds, callers must ensure that the `binding` is not
    // destroyed during a call to `from_python`. The requirements for this are
    // subtle; see the full discussion in the comment for `struct pymb_binding`.
    void* (*from_python)(struct pymb_binding* binding,
                         PyObject* pyobj,
                         uint8_t convert,
                         void (*keep_referenced)(void* ctx, PyObject* obj),
                         void* keep_referenced_ctx);

    // Wrap the C/C++/etc object `cobj` into a Python object using the given
    // return value policy. The type is specified by providing a `pymb_binding*`
    // for some binding that belongs to this framework. `parent` is relevant
    // only if `rvp == pymb_rv_policy_reference_internal`. rvp must be one of
    // the defined enumerators. Returns NULL if the cast is not possible, or
    // a new reference otherwise.
    //
    // A NULL return may leave the Python error indicator set if something
    // specifically describable went wrong during conversion, but is not
    // required to; returning NULL without PyErr_Occurred() should be
    // interpreted as a generic failure to convert `cobj` to a Python object.
    //
    // On free-threaded builds, callers must ensure that the `binding` is not
    // destroyed during a call to `to_python`. The requirements for this are
    // subtle; see the full discussion in the comment for `struct pymb_binding`.
    PyObject* (*to_python)(struct pymb_binding* binding,
                           void* cobj,
                           enum pymb_rv_policy rvp,
                           PyObject* parent);

    // Request that a PyObject reference be dropped, or that a callback
    // be invoked, when `nurse` is destroyed. `nurse` should be an object
    // whose type is bound by this framework. If `cb` is NULL, then
    // `payload` is a PyObject* to decref; otherwise `payload` will
    // be passed as the argument to `cb`. Returns 0 if successful,
    // or -1 and sets the Python error indicator on error.
    //
    // No synchronization is required to call this method.
    int (*keep_alive)(PyObject* nurse, void* payload, void (*cb)(void*));

    // Attempt to translate a C++ exception known to this framework to Python.
    // This should translate only framework-specific exceptions or user-defined
    // exceptions that were registered with the framework, not generic
    // ones such as `std::exception`. If successful, return normally with the
    // Python error indicator set; otherwise, reraise the provided exception.
    // `eptr` should be cast to `const std::exception_ptr* eptr` before use.
    // This function pointer may be NULL if this framework does not provide
    // C++ exception translation.
    //
    // No synchronization is required to call this method.
    void (*translate_exception)(const void* eptr);

    // Notify this framework that some other framework published a new binding.
    // This call will be made after the new binding has been linked into the
    // `pymb_registry::bindings` list.
    //
    // The `pymb_registry::mutex` or GIL will be held when calling this method.
    void (*add_foreign_binding)(struct pymb_binding* binding);

    // Notify this framework that some other framework is about to remove
    // a binding. This call will be made after the binding has been removed
    // from the `pymb_registry::bindings` list.
    //
    // The `pymb_registry::mutex` or GIL will be held when calling this method.
    void (*remove_foreign_binding)(struct pymb_binding* binding);

    // Notify this framework that some other framework came into existence.
    // This call will be made after the new framework has been linked into the
    // `pymb_registry::frameworks` list and before it adds any bindings.
    //
    // The `pymb_registry::mutex` or GIL will be held when calling this method.
    void (*add_foreign_framework)(struct pymb_framework* framework);

    // There is no remove_foreign_framework(); the interpreter has
    // already been finalized at that point, so there's nothing for the
    // callback to do.
};

/*
 * Information about one type binding that belongs to a registered framework.
 *
 * A framework that binds some type and wants to allow other frameworks to
 * work with objects of that type must create a `pymb_binding` structure for
 * the type. This can be allocated in any way that the framework prefers (e.g.,
 * on the heap or within the type object). Once filled out, the binding
 * structure should be passed to `pymb_add_binding()`. If the Python type object
 * underlying the binding is to be deallocated, a `pymb_remove_binding()` call
 * must be made, and the `pymb_binding` structure cannot be deallocated until
 * `pymb_remove_binding()` returns. The call to `pymb_remove_binding()`
 * must occur *during* deallocation of the binding's Python type object, i.e.,
 * at a time when `Py_REFCNT(pytype) == 0` but the storage for `pytype` is not
 * yet eligible to be reused for another object. Many frameworks use a custom
 * metaclass, and can add the call to `pymb_remove_binding()` from the metaclass
 * `tp_dealloc`; those that don't can use a weakref callback on the type object
 * instead. The constraint on destruction timing allows `pymb_try_ref_binding()`
 * to temporarily prevent the binding's destruction by incrementing the type
 * object's reference count.
 *
 * Each Python type object for which a `pymb_binding` exists will have an
 * attribute "__pymetabind_binding__" whose value is a capsule object
 * that contains the `pymb_binding` pointer under the name "pymetabind_binding".
 * The attribute is set during `pymb_add_binding()`. This is provided to allow:
 * - Determining which framework to call for a foreign `keep_alive` operation
 * - Locating `pymb_binding` objects for types written in a different language
 *   than yours (where you can't look up by the `pymb_binding::native_type`),
 *   so that you can work with their contents using non-Python-specific
 *   cross-language support
 * - Extracting the native object from a Python object without being too picky
 *   about what type it is (risky, but maybe you have out-of-band information
 *   that shows it's safe)
 * The preferred mechanism for same-language object access is to maintain a
 * hashtable keyed on `pymb_binding::native_type` and look up the binding for
 * the type you want/have. Compared to reading the capsule, this better
 * supports inheritance, to-Python conversions, and implicit conversions, and
 * it's probably also faster depending on how it's implemented.
 *
 * It is valid for multiple frameworks to claim (in separate bindings) the
 * same C/C++ type, or even the same Python type. (A case where multiple
 * frameworks would bind the same Python type is if one is acting as an
 * extension to the other, such as to support extracting pointers to
 * non-primary base classes when the base framework doesn't think about
 * such things.) If multiple frameworks claim the same Python type, then each
 * new registrant will replace the "__pymetabind_binding__" capsule and there
 * is no way to locate the other bindings from the type object.
 *
 * All fields of this structure are set before it is made visible to other
 * threads and then never changed, so they don't need locking to access.
 * However, on free-threaded builds it is necessary to validate that the type
 * object is not partway through being destroyed before you use the binding,
 * and prevent such destruction from beginning until you're done. To do so,
 * call `pymb_try_ref_binding()`; if it returns false, don't use the binding,
 * else use it and then call `pymb_unref_binding()` when done.
 * (On non-free-threaded builds, these do incref/decref to prevent destruction
 * of the type from starting, but can't fail because there's no *concurrent*
 * destruction hazard.)
 *
 * In order to work with one framework's Python objects of a certain type, other
 * frameworks must be able to locate a `pymb_binding` structure for that type.
 * It is expected that they will maintain their own type-to-binding maps, which
 * they can keep up-to-date via their `pymb_framework::add_foreign_binding` and
 * `pymb_framework::remove_foreign_binding` hooks. It is important to think very
 * carefully about how to design the synchronization for these maps so that
 * lookups do not return pointers to bindings that have been deallocated.
 * The remainder of this comment provides some suggestions.
 *
 * The recommended way to handle synchronization is to protect your type lookup
 * map with a readers/writer lock. In your `remove_foreign_binding` hook,
 * obtain a write lock, and hold it while removing the corresponding entry from
 * the map. Before performing a type lookup, obtain a read lock. If the lookup
 * succeeds, call `pymb_try_ref_binding()` on the resulting binding before
 * you release your read lock. Since the binding structure can't be deallocated
 * until all `remove_foreign_binding` hooks have returned, this scheme provides
 * effective protection. It is important not to hold the read lock while
 * executing arbitrary Python code, since a deadlock would result if the type
 * object is deallocated (requiring a write lock) while the read lock were held.
 * Note that `pymb_framework::from_python` for many popular frameworks is
 * capable of executing arbitrary Python code to perform implicit conversions.
 *
 * The lock on a single shared type lookup map is a contention bottleneck,
 * especially if you don't have a readers/writer lock and wish to get by with
 * an ordinary mutex. To improve performance, you can give each thread its
 * own lookup map, and require `remove_foreign_binding` to update all of them.
 * As long as the per-thread maps are always visited in a consistent order
 * when removing a binding, the splitting shouldn't introduce new deadlocks.
 * Since each thread has a separate mutex for its separate map, contention
 * occurs only when bindings are being added or removed, which is much less
 * common than using them.
 */
struct pymb_binding {
    // Hook by which this structure is linked into the list of
    // `pymb_registry::bindings`
    struct pymb_list_node hook;

    // The framework that provides this binding
    struct pymb_framework* framework;

    // Python type: you will get an instance of this type from a successful
    // call to `framework::from_python()` that passes this binding
    PyTypeObject* pytype;

    // The native identifier for this type in `framework->abi_lang`, if that is
    // a concept that exists in that language. See the documentation of
    // `enum pymb_abi_lang` for specific per-language semantics.
    const void* native_type;

    // The way that this type would be written in `framework->abi_lang` source
    // code, as a NUL-terminated byte string without struct/class/enum words.
    // Examples: "Foo", "Bar::Baz", "std::vector<int, std::allocator<int> >"
    const char* source_name;

    // Pointer that is free for use by the framework, e.g., to point to its
    // own data about this type. If the framework needs more data, it can
    // over-allocate the `pymb_binding` storage and use the space after this.
    void* context;
};

/*
 * Users of non-C/C++ languages are welcome to replicate the logic of these
 * inline functions rather than calling them. Their implementations are
 * considered part of the ABI.
 */

PYMB_FUNC struct pymb_registry* pymb_get_registry();
PYMB_FUNC void pymb_add_framework(struct pymb_registry* registry,
                                  struct pymb_framework* framework);
PYMB_FUNC void pymb_remove_framework(struct pymb_registry* registry,
                                     struct pymb_framework* framework);
PYMB_FUNC void pymb_add_binding(struct pymb_registry* registry,
                                struct pymb_binding* binding);
PYMB_FUNC void pymb_remove_binding(struct pymb_registry* registry,
                                   struct pymb_binding* binding);
PYMB_FUNC int pymb_try_ref_binding(struct pymb_binding* binding);
PYMB_FUNC void pymb_unref_binding(struct pymb_binding* binding);
PYMB_FUNC struct pymb_binding* pymb_get_binding(PyObject* type);

#if !defined(PYMB_DECLS_ONLY)

/*
 * Locate an existing `pymb_registry`, or create a new one if necessary.
 * Returns a pointer to it, or NULL with the CPython error indicator set.
 * This must be called from a module initialization function so that the
 * import lock can provide mutual exclusion.
 */
PYMB_FUNC struct pymb_registry* pymb_get_registry() {
#if defined(PYPY_VERSION)
    PyObject* dict = PyEval_GetBuiltins();
#elif PY_VERSION_HEX < 0x03090000
    PyObject* dict = PyInterpreterState_GetDict(_PyInterpreterState_Get());
#else
    PyObject* dict = PyInterpreterState_GetDict(PyInterpreterState_Get());
#endif
    PyObject* key = PyUnicode_FromString("__pymetabind_registry__");
    if (!dict || !key) {
        Py_XDECREF(key);
        return NULL;
    }
    PyObject* capsule = PyDict_GetItem(dict, key);
    if (capsule) {
        Py_DECREF(key);
        return (struct pymb_registry*) PyCapsule_GetPointer(
                capsule, "pymetabind_registry");
    }
    struct pymb_registry* registry;
    registry = (struct pymb_registry*) calloc(1, sizeof(*registry));
    if (registry) {
        pymb_list_init(&registry->frameworks);
        pymb_list_init(&registry->bindings);
        capsule = PyCapsule_New(registry, "pymetabind_registry", NULL);
        int rv = capsule ? PyDict_SetItem(dict, key, capsule) : -1;
        Py_XDECREF(capsule);
        if (rv != 0) {
            free(registry);
            registry = NULL;
        }
    } else {
        PyErr_NoMemory();
    }
    Py_DECREF(key);
    return registry;
}

/*
 * Add a new framework to the given registry. Makes calls to
 * framework->add_foreign_framework() and framework->add_foreign_binding()
 * for each existing framework/binding in the registry.
 */
PYMB_FUNC void pymb_add_framework(struct pymb_registry* registry,
                                  struct pymb_framework* framework) {
#if defined(Py_GIL_DISABLED) && PY_VERSION_HEX < 0x030e0000
    assert(framework->bindings_usable_forever &&
           "Free-threaded removal of bindings requires PyUnstable_TryIncRef(), "
           "which was added in CPython 3.14");
#endif
    pymb_lock_registry(registry);
    PYMB_LIST_FOREACH(struct pymb_framework*, other, registry->frameworks) {
        // Intern `abi_extra` strings so they can be compared by pointer
        if (other->abi_extra && framework->abi_extra &&
            0 == strcmp(other->abi_extra, framework->abi_extra)) {
            framework->abi_extra = other->abi_extra;
            break;
        }
    }
    pymb_list_append(&registry->frameworks, &framework->hook);
    PYMB_LIST_FOREACH(struct pymb_framework*, other, registry->frameworks) {
        if (other != framework) {
            other->add_foreign_framework(framework);
            framework->add_foreign_framework(other);
        }
    }
    PYMB_LIST_FOREACH(struct pymb_binding*, binding, registry->bindings) {
        if (binding->framework != framework && pymb_try_ref_binding(binding)) {
            framework->add_foreign_binding(binding);
            pymb_unref_binding(binding);
        }
    }
    pymb_unlock_registry(registry);
}

/* Add a new binding to the given registry */
PYMB_FUNC void pymb_add_binding(struct pymb_registry* registry,
                                struct pymb_binding* binding) {
#if defined(Py_GIL_DISABLED) && PY_VERSION_HEX >= 0x030e0000
    PyUnstable_EnableTryIncRef((PyObject *) binding->pytype);
#endif
    PyObject* capsule = PyCapsule_New(binding, "pymetabind_binding", NULL);
    int rv = -1;
    if (capsule) {
        rv = PyObject_SetAttrString((PyObject *) binding->pytype,
                                    "__pymetabind_binding__", capsule);
        Py_DECREF(capsule);
    }
    if (rv != 0) {
        PyErr_WriteUnraisable((PyObject *) binding->pytype);
    }
    pymb_lock_registry(registry);
    pymb_list_append(&registry->bindings, &binding->hook);
    PYMB_LIST_FOREACH(struct pymb_framework*, other, registry->frameworks) {
        if (other != binding->framework) {
            other->add_foreign_binding(binding);
        }
    }
    pymb_unlock_registry(registry);
}

/*
 * Remove a binding from the given registry. This must be called during
 * deallocation of the `binding->pytype`, such that its reference count is
 * zero but still accessible. Once this function returns, you can free the
 * binding structure.
 */
PYMB_FUNC void pymb_remove_binding(struct pymb_registry* registry,
                                   struct pymb_binding* binding) {
    pymb_lock_registry(registry);
    pymb_list_unlink(&binding->hook);
    PYMB_LIST_FOREACH(struct pymb_framework*, other, registry->frameworks) {
        if (other != binding->framework) {
            other->remove_foreign_binding(binding);
        }
    }
    pymb_unlock_registry(registry);
}

/*
 * Increase the reference count of a binding. Return 1 if successful (you can
 * use the binding and must call pymb_unref_binding() when done) or 0 if the
 * binding is being removed and shouldn't be used.
 */
PYMB_FUNC int pymb_try_ref_binding(struct pymb_binding* binding) {
#if defined(Py_GIL_DISABLED)
    if (!binding->framework->bindings_usable_forever) {
#if PY_VERSION_HEX >= 0x030e0000
        return PyUnstable_TryIncRef((PyObject *) binding->pytype);
#else
        // bindings_usable_forever is required on this Python version, and
        // was checked in pymb_add_framework()
        assert(false);
#endif
    }
#else
    Py_INCREF((PyObject *) binding->pytype);
#endif
    return 1;
}

/* Decrease the reference count of a binding. */
PYMB_FUNC void pymb_unref_binding(struct pymb_binding* binding) {
#if defined(Py_GIL_DISABLED)
    if (!binding->framework->bindings_usable_forever) {
#if PY_VERSION_HEX >= 0x030e0000
        Py_DECREF((PyObject *) binding->pytype);
#else
        // bindings_usable_forever is required on this Python version, and
        // was checked in pymb_add_framework()
        assert(false);
#endif
    }
#else
    Py_DECREF((PyObject *) binding->pytype);
#endif
}

/*
 * Return a pointer to a pymb_binding for the Python type `type`, or NULL if
 * none exists.
 */
PYMB_FUNC struct pymb_binding* pymb_get_binding(PyObject* type) {
    PyObject* capsule = PyObject_GetAttrString(type, "__pymetabind_binding__");
    if (capsule == NULL) {
        PyErr_Clear();
        return NULL;
    }
    void* binding = PyCapsule_GetPointer(capsule, "pymetabind_binding");
    Py_DECREF(capsule);
    if (!binding) {
        PyErr_Clear();
    }
    return (struct pymb_binding*) binding;
}

#endif /* defined(PYMB_DECLS_ONLY) */

#if defined(__cplusplus)
}
#endif
