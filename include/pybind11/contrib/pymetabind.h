/*
 * pymetabind.h: definitions for interoperability between different
 *               Python binding frameworks
 *
 * Copy this header file into the implementation of a framework that uses it.
 * This functionality is intended to be used by the framework itself,
 * rather than by users of the framework.
 *
 * This is version 0.3 of pymetabind. Changelog:
 *
 *     Version 0.3: Don't do a Py_DECREF in `pymb_remove_framework` since the
 *      2025-09-15  interpreter might already be finalized at that point.
 *                  Revamp binding lifetime logic. Add `remove_local_binding`
 *                  and `free_local_binding` callbacks.
 *                  Add `pymb_framework::registry` and use it to simplify
 *                  the signatures of `pymb_remove_framework`,
 *                  `pymb_add_binding`, and `pymb_remove_binding`.
 *                  Update `to_python` protocol to be friendlier to
 *                  pybind11 instances with shared/smart holders.
 *                  Remove `pymb_rv_policy_reference_internal`; add
 *                  `pymb_rv_policy_share_ownership`. Change `keep_alive`
 *                  return value convention.
 *
 *     Version 0.2: Use a bitmask for `pymb_framework::flags` and add leak_safe
 *      2025-09-11  flag. Change `translate_exception` to be non-throwing.
 *                  Add casts from PyTypeObject* to PyObject* where needed.
 *                  Fix typo in Py_GIL_DISABLED. Add noexcept to callback types.
 *                  Rename `hook` -> `link` in linked list nodes.
 *                  Use `static inline` linkage in C. Free registry on exit.
 *                  Clear list hooks when adding frameworks/bindings in case
 *                  the user didn't zero-initialize. Avoid abi_extra string
 *                  comparisons if the strings are already pointer-equal.
 *                  Add `remove_foreign_framework` callback.
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
#include <string.h>

#if !defined(PY_VERSION_HEX)
#  error You must include Python.h before this header
#endif

// `inline` in C implies a promise to provide an out-of-line definition
// elsewhere; in C++ it does not.
#ifdef __cplusplus
#  define PYMB_INLINE inline
#else
#  define PYMB_INLINE static inline
#endif

/*
 * There are two ways to use this header file. The default is header-only style,
 * where all functions are defined as `inline` (C++) / `static inline` (C).
 * If you want to emit functions as non-inline, perhaps so you can link against
 * them from non-C/C++ code, then do the following:
 * - In every compilation unit that includes this header, `#define PYMB_FUNC`
 *   first. (The `PYMB_FUNC` macro will be expanded in place of the "inline"
 *   keyword, so you can also use it to add any other declaration attributes
 *   required by your environment.)
 * - In all those compilation units except one, also `#define PYMB_DECLS_ONLY`
 *   before including this header. The definitions will be emitted in the
 *   compilation unit that doesn't request `PYMB_DECLS_ONLY`.
 */
#if !defined(PYMB_FUNC)
#define PYMB_FUNC PYMB_INLINE
#endif

#if defined(__cplusplus)
#define PYMB_NOEXCEPT noexcept
extern "C" {
#else
#define PYMB_NOEXCEPT
#endif

/*
 * Approach used to cast a previously unknown native instance into a Python
 * object. This is similar to `pybind11::return_value_policy` or
 * `nanobind::rv_policy`; some different options are provided than those,
 * but same-named enumerators have the same semantics and values.
 */
enum pymb_rv_policy {
    // Create a Python object that wraps a pointer to a heap-allocated
    // native instance and will destroy and deallocate it (in whatever way
    // is most natural for the target language) when the Python object is
    // destroyed
    pymb_rv_policy_take_ownership = 2,

    // Create a Python object that owns a new native instance created by
    // copying the given one
    pymb_rv_policy_copy = 3,

    // Create a Python object that owns a new native instance created by
    // moving the given one
    pymb_rv_policy_move = 4,

    // Create a Python object that wraps a pointer to a native instance
    // but will not destroy or deallocate it
    pymb_rv_policy_reference = 5,

    // Create a Python object that wraps a pointer to a native instance
    // and will perform a custom action when the Python object is destroyed.
    // The custom action is specified using the first call to keep_alive()
    // after the object is created, and such a call must occur in order for
    // the object to be considered fully initialized.
    pymb_rv_policy_share_ownership = 6,

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

PYMB_INLINE void pymb_list_init(struct pymb_list* list) {
    list->head.prev = list->head.next = &list->head;
}

PYMB_INLINE void pymb_list_unlink(struct pymb_list_node* node) {
    if (node->next) {
        node->next->prev = node->prev;
        node->prev->next = node->next;
        node->next = node->prev = NULL;
    }
}

PYMB_INLINE void pymb_list_append(struct pymb_list* list,
                                  struct pymb_list_node* node) {
    pymb_list_unlink(node);
    struct pymb_list_node* tail = list->head.prev;
    tail->next = node;
    list->head.prev = node;
    node->prev = tail;
    node->next = &list->head;
}

PYMB_INLINE int pymb_list_is_empty(struct pymb_list* list) {
    return list->head.next == &list->head;
}

#define PYMB_LIST_FOREACH(type, name, list)      \
    for (type name = (type) (list).head.next;    \
         name != (type) &(list).head;            \
         name = (type) name->link.next)

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

    // Heap-allocated PyMethodDef for bound type weakref callback
    PyMethodDef* weakref_callback_def;

    // Reserved for future extensions; currently set to 0
    uint16_t reserved;

    // Set to true when the capsule that points to this registry is destroyed
    uint8_t deallocate_when_empty;

#if defined(Py_GIL_DISABLED)
    // Mutex guarding accesses to `frameworks` and `bindings`.
    // On non-free-threading builds, these are guarded by the Python GIL.
    PyMutex mutex;
#endif
};

#if defined(Py_GIL_DISABLED)
PYMB_INLINE void pymb_lock_registry(struct pymb_registry* registry) {
    PyMutex_Lock(&registry->mutex);
}
PYMB_INLINE void pymb_unlock_registry(struct pymb_registry* registry) {
    PyMutex_Unlock(&registry->mutex);
}
#else
PYMB_INLINE void pymb_lock_registry(struct pymb_registry* registry) {
    (void) registry;
}
PYMB_INLINE void pymb_unlock_registry(struct pymb_registry* registry) {
    (void) registry;
}
#endif

struct pymb_binding;

/* Flags for a `pymb_framework` */
enum pymb_framework_flags {
    // Does this framework guarantee that its `pymb_binding` structures remain
    // valid to use for the lifetime of the Python interpreter process once
    // they have been linked into the lists in `pymb_registry`? Setting this
    // flag reduces the number of atomic operations needed to work with
    // this framework's bindings in free-threaded builds.
    pymb_framework_bindings_usable_forever = 0x0001,

    // Does this framework reliably deallocate all of its type and function
    // objects by the time the Python interpreter is finalized, in the absence
    // of bugs in user code? If not, it might cause leaks of other frameworks'
    // types or functions, via attributes or default argument values for
    // this framework's leaked objects. Other frameworks can suppress their
    // leak warnings (if so equipped) when a non-`leak_safe` framework is added.
    pymb_framework_leak_safe = 0x0002,
};

/* Additional results from `pymb_framework::to_python` */
struct pymb_to_python_feedback {
    // Ignored on entry. On exit, set to 1 if the returned Python object
    // was created by the `to_python` call, or zero if it already existed and
    // was simply looked up.
    uint8_t is_new;

    // On entry, indicates whether the caller can control whether the native
    // instance `cobj` passed to `to_python` is destroyed after the conversion:
    // set to 1 if a relocation is allowable or 0 if `cobj` must be destroyed
    // after the call. (This is only relevant when using pymb_rv_policy_move.)
    // On exit, set to 1 if destruction should be inhibited because `*cobj`
    // was relocated into the new instance. Must be left as zero on exit if
    // set to zero on entry.
    uint8_t relocate;
};

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
 * calling `pymb_remove_framework()` and then deallocating the
 * `pymb_framework` structure.
 *
 * All fields of this structure are set before it is made visible to other
 * threads and then never changed, so they don't need locking to access.
 * Some methods require locking or other synchronization to call; see their
 * individual documentation.
 */
struct pymb_framework {
    // Links to the previous and next framework in the list of
    // `pymb_registry::frameworks`. May be modified as other frameworks are
    // added; protected by the `pymb_registry::mutex` in free-threaded builds.
    struct pymb_list_node link;

    // Link to the `pymb_registry` that this framework is registered with.
    // Filled in by `pymb_add_framework()`.
    struct pymb_registry* registry;

    // Human-readable description of this framework, as a NUL-terminated string
    const char* name;

    // Flags for this framework, a combination of `enum pymb_framework_flags`.
    // Undefined flags must be set to zero to allow for future
    // backward-compatible extensions.
    uint16_t flags;

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
    // and must not throw C++ exceptions. They must not be NULL; if a feature
    // is not relevant to your use case, provide a stub that always fails.

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
    // nanobind or loader_life_support in pybind11. The pointer returned by
    // `from_python` may be invalidated once the `keep_referenced` references
    // are dropped. If you're converting a function argument, you should keep
    // any `keep_referenced` references alive until the function returns.
    // If you're converting for some other purpose, you probably want to make
    // a copy of the object to which `from_python`'s return value points before
    // you drop the references.
    //
    // On free-threaded builds, no direct synchronization is required to call
    // this method, but you must ensure the `binding` won't be destroyed during
    // (or before) your call. This generally requires maintaining a continuously
    // attached Python thread state whenever you hold a pointer to `binding`
    // that a concurrent call to your framework's `remove_foreign_binding`
    // method wouldn't be able to clear. See the comment for `pymb_binding`.
    void* (*from_python)(struct pymb_binding* binding,
                         PyObject* pyobj,
                         uint8_t convert,
                         void (*keep_referenced)(void* ctx, PyObject* obj),
                         void* keep_referenced_ctx) PYMB_NOEXCEPT;

    // Wrap the C/C++/etc object `cobj` into a Python object using the given
    // return value policy. The type is specified by providing a `pymb_binding*`
    // for some binding that belongs to this framework.
    //
    // The semantics of this function are as follows:
    // - If there is already a live Python object created by this framework for
    //   this C++ object address and type, it will be returned and the `rvp` is
    //   ignored.
    // - Otherwise, if `rvp == pymb_rv_policy_none`, NULL is returned without
    //   the Python error indicator set.
    // - Otherwise, a new Python object will be created and returned. It will
    //   wrap either the pointer `cobj` or a copy/move of the contents of
    //   `cobj`, depending on the value of `rvp`.
    //
    // Returns a new reference to a Python object, or NULL if not possible.
    // Also sets *feedback to provide additional information about the
    // conversion.
    //
    // After a successful `to_python` call that returns a new instance and
    // used `pymb_rv_policy_share_ownership`, the caller must make a call to
    // `keep_alive` to describe how the shared ownership should be managed.
    //
    // On free-threaded builds, no direct synchronization is required to call
    // this method, but you must ensure the `binding` won't be destroyed during
    // (or before) your call. This generally requires maintaining a continuously
    // attached Python thread state whenever you hold a pointer to `binding`
    // that a concurrent call to your framework's `remove_foreign_binding`
    // method wouldn't be able to clear. See the comment for `pymb_binding`.
    PyObject* (*to_python)(struct pymb_binding* binding,
                           void* cobj,
                           enum pymb_rv_policy rvp,
                           struct pymb_to_python_feedback* feedback) PYMB_NOEXCEPT;

    // Request that a PyObject reference be dropped, or that a callback
    // be invoked, when `nurse` is destroyed. `nurse` should be an object
    // whose type is bound by this framework. If `cb` is NULL, then
    // `payload` is a PyObject* to decref; otherwise `payload` will
    // be passed as the argument to `cb`. Returns 1 if successful,
    // 0 on error. This method may always return 0 if the framework has
    // no better way to do a keep-alive than by creating a weakref;
    // it is expected that the caller can handle creating the weakref.
    //
    // No synchronization is required to call this method.
    int (*keep_alive)(PyObject* nurse,
                      void* payload,
                      void (*cb)(void*)) PYMB_NOEXCEPT;

    // Attempt to translate the native exception `eptr` into a Python exception.
    // If `abi_lang` is C++, then `eptr` should be cast to `std::exception_ptr*`
    // before use; semantics for other languages have not been defined yet. This
    // should translate only framework-specific exceptions or user-defined
    // exceptions that were registered with the framework, not generic ones
    // such as `std::exception`. If translation succeeds, return 1 with the
    // Python error indicator set; otherwise, return 0. An exception may be
    // converted into a different exception by modifying `*eptr` and returning
    // zero. This method may be set to NULL if its framework does not have
    // a concept of exception translation.
    //
    // No synchronization is required to call this method.
    int (*translate_exception)(void* eptr) PYMB_NOEXCEPT;

    // Notify this framework that one of its own bindings is being removed.
    // This will occur synchronously from within a call to
    // `pymb_remove_binding()`. Don't free the binding yet; wait for a later
    // call to `free_local_binding`.
    //
    // The `pymb_registry::mutex` or GIL will be held when calling this method.
    void (*remove_local_binding)(struct pymb_binding* binding) PYMB_NOEXCEPT;

    // Request this framework to free one of its own binding structures.
    // A call to `pymb_remove_binding()` will eventually result in a call to
    // this method, once pymetabind can prove no one is concurrently using the
    // binding anymore.
    //
    // No synchronization is required to call this method.
    void (*free_local_binding)(struct pymb_binding* binding) PYMB_NOEXCEPT;

    // Notify this framework that some other framework published a new binding.
    // This call will be made after the new binding has been linked into the
    // `pymb_registry::bindings` list.
    //
    // The `pymb_registry::mutex` or GIL will be held when calling this method.
    void (*add_foreign_binding)(struct pymb_binding* binding) PYMB_NOEXCEPT;

    // Notify this framework that some other framework is about to remove
    // a binding. This call will be made after the binding has been removed
    // from the `pymb_registry::bindings` list.
    //
    // The `pymb_registry::mutex` or GIL will be held when calling this method.
    void (*remove_foreign_binding)(struct pymb_binding* binding) PYMB_NOEXCEPT;

    // Notify this framework that some other framework came into existence.
    // This call will be made after the new framework has been linked into the
    // `pymb_registry::frameworks` list and before it adds any bindings.
    //
    // The `pymb_registry::mutex` or GIL will be held when calling this method.
    void (*add_foreign_framework)(struct pymb_framework* framework) PYMB_NOEXCEPT;

    // Notify this framework that some other framework is being destroyed.
    // This call will be made after the framework has been removed from the
    // `pymb_registry::frameworks` list.
    //
    // This can only occur during interpreter finalization, so no
    // synchronization is required. It might occur very late in interpreter
    // finalization, such as from a Py_AtExit handler, so it shouldn't
    // execute Python code.
    void (*remove_foreign_framework)(struct pymb_framework* framework) PYMB_NOEXCEPT;
};

/*
 * Information about one type binding that belongs to a registered framework.
 *
 * ### Creating bindings
 *
 * A framework that binds some type and wants to allow other frameworks to
 * work with objects of that type must create a `pymb_binding` structure for
 * the type. This can be allocated in any way that the framework prefers (e.g.,
 * on the heap or within the type object). Any fields without a meaningful
 * value must be zero-filled. Once filled out, the binding structure should be
 * passed to `pymb_add_binding()`. This will advertise the binding to other
 * frameworks' `add_foreign_binding` hooks. It also creates a capsule object
 * that points to the `pymb_binding` structure, and stores this capsule in the
 * bound type's dict as the attribute "__pymetabind_binding__".
 * The intended use of both of these is discussed later on in this comment.
 *
 * ### Removing bindings
 *
 * From a user perspective, a binding can be removed for either of two reasons:
 *
 * - its capsule was destroyed, such as by `del MyType.__pymetabind_binding__`
 * - its type object is being finalized
 *
 * These both result in a call to `pymb_remove_binding()` that begins the
 * removal process, but you should not call that function yourself, except
 * from a metatype `tp_finalize` as described below. Some time after the call
 * to `pymb_remove_binding()`, pymetabind will call the binding's framework's
 * `free_local_binding` hook to indicate that it's safe to actually free the
 * `pymb_binding` structure.
 *
 * By default, pymetabind detects the finalization of a binding's type object
 * by creating a weakref to the type object with an appropriate callback. This
 * works fine, but requires several objects to be allocated, so it is not ideal
 * from a memory overhead perspective. If you control the bound type's metatype,
 * you can reduce this overhead by modifying the metatype's `tp_finalize` slot
 * to call `pymb_remove_binding()`. If you tell pymetabind that you have done
 * so, using the `tp_finalize_will_remove` argument to `pymb_add_binding()`,
 * then pymetabind won't need to create the weakref and its callback.
 *
 * ### Removing bindings: the gory details
 *
 * The implementation of the removal process is somewhat complex in order to
 * protect other threads that might be concurrently using a binding in
 * free-threaded builds. `pymb_remove_binding()` stops new uses of the binding
 * from beginning, by notifying other frameworks' `remove_foreign_binding`
 * hooks and changing the binding capsule so `pymb_get_binding()` won't work.
 * Existing uses might be ongoing though, so we must wait for them to complete
 * before freeing the binding structure. The technique we choose is to wait for
 * the next (or current) garbage collection to finish. GC stops all threads
 * before it scans the heap. An attached thread state (one that can call
 * CPython API functions) can't be stopped without its consent, so GC will
 * wait for it to detach. A thread state can only become detached explicitly
 * (e.g. Py_BEGIN_ALLOW_THREADS) or in the bytecode interpreter. As long as
 * foreign frameworks don't hold `pymb_binding` pointers across calls into
 * the bytecode interpreter in places their `remove_foreign_binding` hook
 * can't see, this technique avoids use-after-free without introducing any
 * contention on a shared atomic in the binding object.
 *
 * One pleasant aspect of this scheme: due to their use of deferred reference
 * counting, type objects in free-threaded Python can only be freed during a
 * GC pass. There is even a stop-all-threads (to check for resurrected objects)
 * in between when GC executes finalizers and when it actually destroys the
 * garbage. This winds up letting us obtain the "wait for next GC pause before
 * freeing the binding" behavior very cheaply when the binding is being removed
 * due to the deletion of its type.
 *
 * On non-free-threaded Python builds, none of the above is a concern, and
 * `pymb_remove_binding()` can synchronously free the `pymb_binding` structure.
 *
 * ### Keeping track of other frameworks' bindings
 *
 * In order to work with Python objects bound by another framework, yours
 * must be able to locate a `pymb_binding` structure for that type. It is
 * anticipated that most frameworks will maintain their own private
 * type-to-binding maps, which they can keep up-to-date via their
 * `add_foreign_binding` and `remove_foreign_binding` hooks. It is important
 * to think carefully about how to design the synchronization for these maps
 * so that lookups do not return pointers to bindings that may have been
 * deallocated. The remainder of this section provides some suggestions.
 *
 * The recommended way to handle synchronization is to protect your type lookup
 * map with a readers/writer lock. In your `remove_foreign_binding` hook,
 * obtain a write lock, and hold it while removing the corresponding entry from
 * the map. Before performing a type lookup, obtain a read lock. If the lookup
 * succeeds, you can release the read lock and (due to the two-phase removal
 * process described above) continue to safely use the binding for as long as
 * your Python thread state remains attached. It is important not to hold the
 * read lock while executing arbitrary Python code, since a deadlock would
 * result if the binding were removed (requiring a write lock) while the read
 * lock were held. Note that `pymb_framework::from_python` for many popular
 * frameworks can execute arbitrary Python code to perform implicit conversions.
 *
 * If you're trying multiple bindings for an operation, one option is to copy
 * all their pointers to temporary storage before releasing the read lock.
 * (While concurrent updates may modify the data structure, the pymb_binding
 * structures it points to will remain valid for long enough.) If you prefer
 * to avoid the copy by unlocking for each attempt and then relocking to
 * advance to the next binding, be sure to consider the possibility that your
 * iterator might have been invalidated due to a concurrent update while you
 * weren't holding the lock.
 *
 * The lock on a single shared type lookup map is a contention bottleneck,
 * especially if you don't have a readers/writer lock and wish to get by with
 * an ordinary mutex. To improve performance, you can give each thread its
 * own lookup map, and require `remove_foreign_binding` to update all of them.
 * As long as the per-thread maps are always visited in a consistent order
 * when removing a binding, the splitting shouldn't introduce new deadlocks.
 * Since each thread can have a separate mutex for its separate map, contention
 * occurs only when bindings are being added or removed, which is much less
 * common than using them.
 *
 * ### Using the binding capsule
 *
 * Each Python type object for which a `pymb_binding` exists will have an
 * attribute "__pymetabind_binding__" whose value is a capsule object
 * that contains the `pymb_binding` pointer under the name "pymetabind_binding".
 * The attribute is set during `pymb_add_binding()`, and is used by
 * `pymb_get_binding()` to map a type object to a binding. The capsule allows:
 *
 * - Determining which framework to call for a foreign `keep_alive` operation
 *
 * - Locating `pymb_binding` objects for types written in a different language
 *   than yours (where you can't look up by the `pymb_binding::native_type`),
 *   so that you can work with their contents using non-Python-specific
 *   cross-language support
 *
 * - Extracting the native object from a Python object without being too picky
 *   about what type it is (risky, but maybe you have out-of-band information
 *   that shows it's safe)
 *
 * The preferred mechanism for same-language object access is to maintain a
 * hashtable keyed on `pymb_binding::native_type` and look up the binding for
 * the type you want/have. Compared to reading the capsule, this better
 * supports inheritance, to-Python conversions, and implicit conversions, and
 * it's probably also faster depending on how it's implemented.
 *
 * ### Types with multiple bindings
 *
 * It is valid for multiple frameworks to claim (in separate bindings) the
 * same C/C++ type. This supports cases where a common vocabulary type is
 * bound separately in mulitple extensions in the same process. Frameworks
 * are encouraged to try all registered bindings for the target type when
 * they perform from-Python conversions.
 *
 * If multiple frameworks claim the same Python type, the last one will
 * typically win, since there is only one "__pymetabind_binding__" attribute
 * on the type object and a binding is removed when its capsule is no longer
 * referenced. If you're trying to do something unusual like wrapping another
 * framework's binding to provide additional features, you can stash the
 * extra binding(s) under a different attribute name. pymetabind never uses
 * the "__pymetabind_binding__" attribute to locate the binding for its own
 * purposes; it's used only to fulfill calls to `pymb_get_binding()`.
 *
 * ### Synchronization
 *
 * Most fields of this structure are set before it is made visible to other
 * threads and then never changed, so they don't need locking to access. The
 * `link` and `capsule` are protected by the registry lock.
 */
struct pymb_binding {
    // Links to the previous and next bindings in the list of
    // `pymb_registry::bindings`
    struct pymb_list_node link;

    // The framework that provides this binding
    struct pymb_framework* framework;

    // Borrowed reference to the capsule object that refers to this binding.
    // Becomes NULL in pymb_remove_binding().
    PyObject* capsule;

    // Python type: you will get an instance of this type from a successful
    // call to `framework::from_python()` that passes this binding
    PyTypeObject* pytype;

    // Strong reference to a weakref to `pytype`; its callback will prompt
    // us to remove the binding. May be NULL if Py_TYPE(pytype)->tp_finalize
    // will take care of that.
    PyObject* pytype_wr;

    // The native identifier for this type in `framework->abi_lang`, if that is
    // a concept that exists in that language. See the documentation of
    // `enum pymb_abi_lang` for specific per-language semantics.
    const void* native_type;

    // The way that this type would be written in `framework->abi_lang` source
    // code, as a NUL-terminated byte string without struct/class/enum words.
    // If `framework->abi_lang` uses name mangling, this is the demangled,
    // human-readable name. C++ users should note that the result of
    // `typeid(x).name()` will need platform-specific alteration to produce one.
    // Examples: "Foo", "Bar::Baz", "std::vector<int, std::allocator<int> >"
    const char* source_name;

    // Pointer that is free for use by the framework, e.g., to point to its
    // own data about this type. If the framework needs more data, it can
    // over-allocate the `pymb_binding` storage and use the space after this.
    void* context;
};

PYMB_FUNC struct pymb_registry* pymb_get_registry();
PYMB_FUNC void pymb_add_framework(struct pymb_registry* registry,
                                  struct pymb_framework* framework);
PYMB_FUNC void pymb_remove_framework(struct pymb_framework* framework);
PYMB_FUNC void pymb_add_binding(struct pymb_binding* binding,
                                int tp_finalize_will_remove);
PYMB_FUNC void pymb_remove_binding(struct pymb_binding* binding);
PYMB_FUNC struct pymb_binding* pymb_get_binding(PyObject* type);

#if !defined(PYMB_DECLS_ONLY)

PYMB_INLINE void pymb_registry_free(struct pymb_registry* registry) {
    assert(pymb_list_is_empty(&registry->bindings) &&
           "some framework was removed before its bindings");
    free(registry->weakref_callback_def);
    free(registry);
}

PYMB_FUNC void pymb_registry_capsule_destructor(PyObject* capsule) {
    struct pymb_registry* registry =
            (struct pymb_registry*) PyCapsule_GetPointer(
                    capsule, "pymetabind_registry");
    if (!registry) {
        PyErr_WriteUnraisable(capsule);
        return;
    }
    registry->deallocate_when_empty = 1;
    if (pymb_list_is_empty(&registry->frameworks)) {
        pymb_registry_free(registry);
    }
}

PYMB_FUNC PyObject* pymb_weakref_callback(PyObject* self, PyObject* weakref) {
    // self is bound using PyCFunction_New to refer to a capsule that contains
    // the binding pointer (not the binding->capsule; this one has no dtor).
    // `weakref` is the weakref (to the bound type) that expired.
    if (!PyWeakref_CheckRefExact(weakref) || !PyCapsule_CheckExact(self)) {
        PyErr_BadArgument();
        return NULL;
    }
    struct pymb_binding* binding =
            (struct pymb_binding*) PyCapsule_GetPointer(
                    self, "pymetabind_binding");
    if (!binding) {
        return NULL;
    }
    pymb_remove_binding(binding);
    Py_RETURN_NONE;
}

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
        registry->deallocate_when_empty = 0;

        // C doesn't allow inline functions to declare static variables,
        // so allocate this on the heap
        PyMethodDef* def = (PyMethodDef*) calloc(1, sizeof(PyMethodDef));
        if (!def) {
            free(registry);
            PyErr_NoMemory();
            Py_DECREF(key);
            return NULL;
        }
        def->ml_name = "pymetabind_weakref_callback";
        def->ml_meth = pymb_weakref_callback;
        def->ml_flags = METH_O;
        def->ml_doc = NULL;
        registry->weakref_callback_def = def;

        // Attach a destructor so the registry memory is released at teardown
        capsule = PyCapsule_New(registry, "pymetabind_registry",
                                pymb_registry_capsule_destructor);
        if (!capsule) {
            free(registry);
        } else if (PyDict_SetItem(dict, key, capsule) == -1) {
            registry = NULL; // will be deallocated by capsule destructor
        }
        Py_XDECREF(capsule);
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
    // Defensive: ensure hook is clean before first list insertion to avoid UB
    framework->link.next = NULL;
    framework->link.prev = NULL;
    framework->registry = registry;
    pymb_lock_registry(registry);
    PYMB_LIST_FOREACH(struct pymb_framework*, other, registry->frameworks) {
        // Intern `abi_extra` strings so they can be compared by pointer
        if (other->abi_extra && framework->abi_extra &&
            (other->abi_extra == framework->abi_extra ||
             strcmp(other->abi_extra, framework->abi_extra) == 0)) {
            framework->abi_extra = other->abi_extra;
            break;
        }
    }
    pymb_list_append(&registry->frameworks, &framework->link);
    PYMB_LIST_FOREACH(struct pymb_framework*, other, registry->frameworks) {
        if (other != framework) {
            other->add_foreign_framework(framework);
            framework->add_foreign_framework(other);
        }
    }
    PYMB_LIST_FOREACH(struct pymb_binding*, binding, registry->bindings) {
        if (binding->framework != framework) {
            framework->add_foreign_binding(binding);
        }
    }
    pymb_unlock_registry(registry);
}

/*
 * Remove a framework from the registry it was added to.
 *
 * This may only be called during Python interpreter finalization. Rationale:
 * other frameworks might be maintaining an entry for the removed one in their
 * exception translator lists, and supporting concurrent removal of exception
 * translators would add undesirable synchronization overhead to the handling
 * of every exception. At finalization time there are no more threads.
 *
 * Once this function returns, you can free the framework structure.
 *
 * If a framework never removes itself, it must not claim to be `leak_safe`.
 */
PYMB_FUNC void pymb_remove_framework(struct pymb_framework* framework) {
    struct pymb_registry* registry = framework->registry;

    // No need for registry lock/unlock since there are no more threads
    pymb_list_unlink(&framework->link);
    PYMB_LIST_FOREACH(struct pymb_framework*, other, registry->frameworks) {
        other->remove_foreign_framework(framework);
    }

    // Destroy registry if capsule is gone and this was the last framework
    if (registry->deallocate_when_empty &&
        pymb_list_is_empty(&registry->frameworks)) {
        pymb_registry_free(registry);
    }
}

PYMB_FUNC void pymb_binding_capsule_remove(PyObject* capsule) {
    struct pymb_binding* binding =
            (struct pymb_binding*) PyCapsule_GetPointer(
                    capsule, "pymetabind_binding");
    if (!binding) {
        PyErr_WriteUnraisable(capsule);
        return;
    }
    pymb_remove_binding(binding);
}

/*
 * Add a new binding for `binding->framework`. If `tp_finalize_will_remove` is
 * nonzero, the caller guarantees that `Py_TYPE(binding->pytype).tp_finalize`
 * will call `pymb_remove_binding()`; this saves some allocations compared
 * to pymetabind needing to figure out when the type is destroyed on its own.
 * See the comment on `pymb_binding` for more details.
 */
PYMB_FUNC void pymb_add_binding(struct pymb_binding* binding,
                                int tp_finalize_will_remove) {
    // Defensive: ensure hook is clean before first list insertion to avoid UB
    binding->link.next = NULL;
    binding->link.prev = NULL;

    binding->pytype_wr = NULL;
    binding->capsule = NULL;

    struct pymb_registry* registry = binding->framework->registry;
    if (!tp_finalize_will_remove) {
        // Different capsule than the binding->capsule, so that the callback
        // doesn't keep the binding alive
        PyObject* sub_capsule = PyCapsule_New(binding, "pymetabind_binding",
                                              NULL);
        if (!sub_capsule) {
            goto error;
        }
        PyObject* callback = PyCFunction_New(registry->weakref_callback_def,
                                             sub_capsule);
        Py_DECREF(sub_capsule); // ownership transferred to callback
        if (!callback) {
            goto error;
        }
        binding->pytype_wr = PyWeakref_NewRef((PyObject *) binding->pytype,
                                              callback);
        Py_DECREF(callback); // ownership transferred to weakref
        if (!binding->pytype_wr) {
            goto error;
        }
    } else {
#if defined(Py_GIL_DISABLED)
        // No callback needed in this case, but we still do need the weakref
        // so that pymb_remove_binding() can tell if the type is being
        // finalized or not.
        binding->pytype_wr = PyWeakref_NewRef((PyObject *) binding->pytype,
                                              NULL);
        if (!binding->pytype_wr) {
            goto error;
        }
#endif
    }

    binding->capsule = PyCapsule_New(binding, "pymetabind_binding",
                                     pymb_binding_capsule_remove);
    if (!binding->capsule) {
        goto error;
    }
    if (PyObject_SetAttrString((PyObject *) binding->pytype,
                               "__pymetabind_binding__",
                               binding->capsule) != 0) {
        Py_CLEAR(binding->capsule);
        goto error;
    }
    Py_DECREF(binding->capsule); // keep only a borrowed reference

    pymb_lock_registry(registry);
    pymb_list_append(&registry->bindings, &binding->link);
    PYMB_LIST_FOREACH(struct pymb_framework*, other, registry->frameworks) {
        if (other != binding->framework) {
            other->add_foreign_binding(binding);
        }
    }
    pymb_unlock_registry(registry);
    return;

  error:
    PyErr_WriteUnraisable((PyObject *) binding->pytype);
    Py_XDECREF(binding->pytype_wr);
    binding->framework->free_local_binding(binding);
}

#if defined(Py_GIL_DISABLED)
PYMB_FUNC void pymb_binding_capsule_destroy(PyObject* capsule) {
    struct pymb_binding* binding =
            (struct pymb_binding*) PyCapsule_GetPointer(
                    capsule, "pymetabind_binding");
    if (!binding) {
        PyErr_WriteUnraisable(capsule);
        return;
    }
    Py_CLEAR(binding->pytype_wr);
    binding->framework->free_local_binding(binding);
}
#endif

/*
 * Remove a binding from the registry it was added to. Don't call this yourself,
 * except from the tp_finalize slot of a binding's type's metatype.
 * The user-servicable way to remove a binding from a still-alive type is to
 * delete the capsule. The binding structure will eventually be freed by calling
 * `binding->framework->free_local_binding(binding)`.
 */
PYMB_FUNC void pymb_remove_binding(struct pymb_binding* binding) {
    struct pymb_registry* registry = binding->framework->registry;

    // Since we need to obtain it anyway, use the registry lock to serialize
    // concurrent attempts to remove the same binding
    pymb_lock_registry(registry);
    if (!binding->capsule) {
        // Binding was concurrently removed from multiple places; the first
        // one to get the registry lock wins.
        pymb_unlock_registry(registry);
        return;
    }

#if defined(Py_GIL_DISABLED)
    // Determine if binding->pytype is still fully alive (not yet started
    // finalizing). If so, it can't die until the next GC cycle, so freeing
    // the binding at the next GC is safe.
    PyObject* pytype_strong = NULL;
    if (PyWeakref_GetRef(binding->pytype_wr, &pytype_strong) == -1) {
        // If something's wrong with the weakref, leave pytype_strong set to
        // NULL in order to conservatively assume the type is finalizing.
        // This will leak the binding struct until the type object is destroyed.
        PyErr_WriteUnraisable((PyObject *) binding->pytype);
    }
#endif

    // Clear the existing capsule's destructor so we don't have to worry about
    // it firing after the pymb_binding struct has actually been freed.
    // Note we can safely assume the capsule hasn't been freed yet, even
    // though it might be mid-destruction. (Proof: Its destructor calls
    // this function, which cannot complete until it acquires the lock we
    // currently hold. If the destructor completed already, we would have bailed
    // out above upon noticing capsule was already NULL.)
    if (PyCapsule_SetDestructor(binding->capsule, NULL) != 0) {
        PyErr_WriteUnraisable((PyObject *) binding->pytype);
    }

    // Mark this binding as being in the process of being destroyed.
    binding->capsule = NULL;

    // If weakref hasn't fired yet, we don't need it anymore. Destroying it
    // ensures it won't fire after the binding struct has been freed.
    Py_CLEAR(binding->pytype_wr);

    pymb_list_unlink(&binding->link);
    binding->framework->remove_local_binding(binding);
    PYMB_LIST_FOREACH(struct pymb_framework*, other, registry->frameworks) {
        if (other != binding->framework) {
            other->remove_foreign_binding(binding);
        }
    }
    pymb_unlock_registry(registry);

#if !defined(Py_GIL_DISABLED)
    // On GIL builds, there's no need to delay deallocation
    binding->framework->free_local_binding(binding);
#else
    // Create a new capsule to manage the actual freeing
    PyObject* capsule_destroy = PyCapsule_New(binding,
                                              "pymetabind_binding",
                                              pymb_binding_capsule_destroy);
    if (!capsule_destroy) {
        // Just leak the binding if we can't set up the capsule
        PyErr_WriteUnraisable((PyObject *) binding->pytype);
    } else if (pytype_strong) {
        // Type still alive -> embed the capsule in a cycle so it lasts until
        // next GC. (The type will live at least that long.)
        PyObject* list = PyList_New(2);
        if (!list) {
            PyErr_WriteUnraisable((PyObject *) binding->pytype);
            // leak the capsule and therefore the binding
        } else {
            PyList_SetItem(list, 0, capsule_destroy);
            PyList_SetItem(list, 1, list);
            // list is now referenced only by itself and will be GCable
        }
    } else {
        // Type is dying -> destroy the capsule when the type is destroyed.
        // Since the type's weakrefs were already cleared, any weakref we add
        // now won't fire until the type's tp_dealloc. We reuse our existing
        // weakref callback for convenience; the call that it makes to
        // pymb_remove_binding() will be a no-op, but after it fires,
        // the capsule destructor will do the freeing we desire.
        PyObject* callback = PyCFunction_New(registry->weakref_callback_def,
                                             capsule_destroy);
        if (!callback) {
            PyErr_WriteUnraisable((PyObject *) binding->pytype);
            // leak the capsule and therefore the binding
        } else {
            Py_DECREF(capsule_destroy); // ownership transferred to callback
            binding->pytype_wr = PyWeakref_NewRef((PyObject *) binding->pytype,
                                                  callback);
            Py_DECREF(callback); // ownership transferred to weakref
            if (!binding->pytype_wr) {
                PyErr_WriteUnraisable((PyObject *) binding->pytype);
            }
        }
    }
    Py_XDECREF(pytype_strong);
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
