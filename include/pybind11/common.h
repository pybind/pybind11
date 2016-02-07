/*
    pybind11/common.h -- Basic macros

    Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#if !defined(NAMESPACE_BEGIN)
#  define NAMESPACE_BEGIN(name) namespace name {
#endif
#if !defined(NAMESPACE_END)
#  define NAMESPACE_END(name) }
#endif

#if !defined(PYBIND11_EXPORT)
#  if defined(WIN32) || defined(_WIN32)
#    define PYBIND11_EXPORT __declspec(dllexport)
#  else
#    define PYBIND11_EXPORT __attribute__ ((visibility("default")))
#  endif
#endif

#if defined(_MSC_VER)
#  define PYBIND11_NOINLINE __declspec(noinline)
#else
#  define PYBIND11_NOINLINE __attribute__ ((noinline))
#endif

#define PYBIND11_VERSION_MAJOR 1
#define PYBIND11_VERSION_MINOR 3

/// Include Python header, disable linking to pythonX_d.lib on Windows in debug mode
#if defined(_MSC_VER)
#  define HAVE_ROUND
#  pragma warning(push)
#  pragma warning(disable: 4510 4610 4512 4005)
#  if _DEBUG
#    define _DEBUG_MARKER
#    undef _DEBUG
#  endif
#endif

#include <Python.h>
#include <frameobject.h>

#ifdef isalnum
#  undef isalnum
#  undef isalpha
#  undef islower
#  undef isspace
#  undef isupper
#  undef tolower
#  undef toupper
#endif

#if defined(_MSC_VER)
#  if defined(_DEBUG_MARKER)
#    define _DEBUG
#  undef _DEBUG_MARKER
#endif
#  pragma warning(pop)
#endif

#include <vector>
#include <string>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <typeindex>

#if PY_MAJOR_VERSION >= 3 /// Compatibility macros for various Python versions
#define PYBIND11_INSTANCE_METHOD_NEW(ptr, class_) PyInstanceMethod_New(ptr)
#define PYBIND11_BYTES_CHECK PyBytes_Check
#define PYBIND11_BYTES_FROM_STRING PyBytes_FromString
#define PYBIND11_BYTES_FROM_STRING_AND_SIZE PyBytes_FromStringAndSize
#define PYBIND11_BYTES_AS_STRING_AND_SIZE PyBytes_AsStringAndSize
#define PYBIND11_BYTES_AS_STRING PyBytes_AsString
#define PYBIND11_LONG_CHECK(o) PyLong_Check(o)
#define PYBIND11_LONG_AS_LONGLONG(o) PyLong_AsLongLong(o)
#define PYBIND11_LONG_AS_UNSIGNED_LONGLONG(o) PyLong_AsUnsignedLongLong(o)
#define PYBIND11_BYTES_NAME "bytes"
#define PYBIND11_STRING_NAME "str"
#define PYBIND11_SLICE_OBJECT PyObject
#define PYBIND11_FROM_STRING PyUnicode_FromString
#define PYBIND11_OB_TYPE(ht_type) (ht_type).ob_base.ob_base.ob_type
#define PYBIND11_PLUGIN_IMPL(name) \
    extern "C" PYBIND11_EXPORT PyObject *PyInit_##name()
#else
#define PYBIND11_INSTANCE_METHOD_NEW(ptr, class_) PyMethod_New(ptr, nullptr, class_)
#define PYBIND11_BYTES_CHECK PyString_Check
#define PYBIND11_BYTES_FROM_STRING PyString_FromString
#define PYBIND11_BYTES_FROM_STRING_AND_SIZE PyString_FromStringAndSize
#define PYBIND11_BYTES_AS_STRING_AND_SIZE PyString_AsStringAndSize
#define PYBIND11_BYTES_AS_STRING PyString_AsString
#define PYBIND11_LONG_CHECK(o) (PyInt_Check(o) || PyLong_Check(o))
#define PYBIND11_LONG_AS_LONGLONG(o) (PyInt_Check(o) ? (long long) PyLong_AsLong(o) : PyLong_AsLongLong(o))
#define PYBIND11_LONG_AS_UNSIGNED_LONGLONG(o) (PyInt_Check(o) ? (unsigned long long) PyLong_AsUnsignedLong(o) : PyLong_AsUnsignedLongLong(o))
#define PYBIND11_BYTES_NAME "str"
#define PYBIND11_STRING_NAME "unicode"
#define PYBIND11_SLICE_OBJECT PySliceObject
#define PYBIND11_FROM_STRING PyString_FromString
#define PYBIND11_OB_TYPE(ht_type) (ht_type).ob_type
#define PYBIND11_PLUGIN_IMPL(name) \
    extern "C" PYBIND11_EXPORT PyObject *init##name()
#endif

#define PYBIND11_TRY_NEXT_OVERLOAD ((PyObject *) 1) // special failure return code

#define PYBIND11_PLUGIN(name) \
    static PyObject *pybind11_init(); \
    PYBIND11_PLUGIN_IMPL(name) { \
        try { \
            return pybind11_init(); \
        } catch (const std::exception &e) { \
            PyErr_SetString(PyExc_ImportError, e.what()); \
            return nullptr; \
        } \
    } \
    PyObject *pybind11_init()


NAMESPACE_BEGIN(pybind11)

typedef Py_ssize_t ssize_t;

/// Approach used to cast a previously unknown C++ instance into a Python object
enum class return_value_policy : int {
    /** Automatic: copy objects returned as values and take ownership of objects
        returned as pointers */
    automatic = 0,
    /** Reference the object and take ownership. Python will call the
        destructor and delete operator when the reference count reaches zero */
    take_ownership,
    /** Reference the object, but do not take ownership (dangerous when C++ code
        deletes it and Python still has a nonzero reference count) */
    reference,
    /** Reference the object, but do not take ownership. The object is considered
        be owned by the C++ instance whose method or property returned it. The
        Python object will increase the reference count of this 'parent' by 1 */
    reference_internal,
    /// Create a new copy of the returned object, which will be owned by Python
    copy
};

/// Format strings for basic number types
template <typename type> struct format_descriptor { };
#define PYBIND11_DECL_FMT(t, n) template<> struct format_descriptor<t> { static std::string value() { return n; }; };
PYBIND11_DECL_FMT(int8_t,  "b"); PYBIND11_DECL_FMT(uint8_t,  "B"); PYBIND11_DECL_FMT(int16_t, "h"); PYBIND11_DECL_FMT(uint16_t, "H");
PYBIND11_DECL_FMT(int32_t, "i"); PYBIND11_DECL_FMT(uint32_t, "I"); PYBIND11_DECL_FMT(int64_t, "q"); PYBIND11_DECL_FMT(uint64_t, "Q");
PYBIND11_DECL_FMT(float,   "f"); PYBIND11_DECL_FMT(double,   "d"); PYBIND11_DECL_FMT(bool,    "?");

/// Information record describing a Python buffer object
struct buffer_info {
    void *ptr;                   // Pointer to the underlying storage
    size_t itemsize;             // Size of individual items in bytes
    size_t size;                 // Total number of entries
    std::string format;          // For homogeneous buffers, this should be set to format_descriptor<T>::value
    int ndim;                    // Number of dimensions
    std::vector<size_t> shape;   // Shape of the tensor (1 entry per dimension)
    std::vector<size_t> strides; // Number of entries between adjacent entries (for each per dimension)

    buffer_info(void *ptr, size_t itemsize, const std::string &format, int ndim,
                const std::vector<size_t> &shape, const std::vector<size_t> &strides)
        : ptr(ptr), itemsize(itemsize), size(1), format(format),
          ndim(ndim), shape(shape), strides(strides) {
        for (int i=0; i<ndim; ++i) size *= shape[i];
    }

    buffer_info(Py_buffer *view)
        : ptr(view->buf), itemsize(view->itemsize), size(1), format(view->format),
          ndim(view->ndim), shape(view->ndim), strides(view->ndim), view(view) {
        for (int i = 0; i < view->ndim; ++i) {
            shape[i] = (size_t) view->shape[i];
            strides[i] = (size_t) view->strides[i];
            size *= shape[i];
        }
    }

    ~buffer_info() {
        if (view) { PyBuffer_Release(view); delete view; }
    }
private:
    Py_buffer *view = nullptr;
};

NAMESPACE_BEGIN(detail)

inline std::string error_string();

/// Core part of the 'instance' type which POD (needed to be able to use 'offsetof')
template <typename type> struct instance_essentials {
    PyObject_HEAD
    type *value;
    PyObject *parent;
    PyObject *weakrefs;
    bool owned : 1;
    bool constructed : 1;
};

/// PyObject wrapper around generic types, includes a special holder type that is responsible for lifetime management
template <typename type, typename holder_type = std::unique_ptr<type>> struct instance : instance_essentials<type> {
    holder_type holder;
};

struct overload_hash {
    inline std::size_t operator()(const std::pair<const PyObject *, const char *>& v) const {
        size_t value = std::hash<const void *>()(v.first);
        value ^= std::hash<const void *>()(v.second)  + 0x9e3779b9 + (value<<6) + (value>>2);
        return value;
    }
};

/// Internal data struture used to track registered instances and types
struct internals {
    std::unordered_map<std::type_index, void*> registered_types_cpp; // std::type_index -> type_info
    std::unordered_map<const void *, void*> registered_types_py;     // PyTypeObject* -> type_info
    std::unordered_map<const void *, void*> registered_instances;    // void * -> PyObject*
    std::unordered_set<std::pair<const PyObject *, const char *>, overload_hash> inactive_overload_cache;
};

/// Return a reference to the current 'internals' information
inline internals &get_internals();

/// Index sequence for convenient template metaprogramming involving tuples
template<size_t ...> struct index_sequence  { };
template<size_t N, size_t ...S> struct make_index_sequence : make_index_sequence <N - 1, N - 1, S...> { };
template<size_t ...S> struct make_index_sequence <0, S...> { typedef index_sequence<S...> type; };

/// Strip the class from a method type
template <typename T> struct remove_class {};
template <typename C, typename R, typename... A> struct remove_class<R (C::*)(A...)> { typedef R type(A...); };
template <typename C, typename R, typename... A> struct remove_class<R (C::*)(A...) const> { typedef R type(A...); };

/// Helper template to strip away type modifiers
template <typename T> struct intrinsic_type                       { typedef T type; };
template <typename T> struct intrinsic_type<const T>              { typedef typename intrinsic_type<T>::type type; };
template <typename T> struct intrinsic_type<T*>                   { typedef typename intrinsic_type<T>::type type; };
template <typename T> struct intrinsic_type<T&>                   { typedef typename intrinsic_type<T>::type type; };
template <typename T> struct intrinsic_type<T&&>                  { typedef typename intrinsic_type<T>::type type; };
template <typename T, size_t N> struct intrinsic_type<const T[N]> { typedef typename intrinsic_type<T>::type type; };
template <typename T, size_t N> struct intrinsic_type<T[N]>       { typedef typename intrinsic_type<T>::type type; };

/** \brief SFINAE helper class to check if a copy constructor is usable (in contrast to 
 * std::is_copy_constructible, this class also checks if the 'new' operator is accessible */
template <typename T>  struct is_copy_constructible {
    template <typename T2> static std::true_type test(decltype(new T2(std::declval<typename std::add_lvalue_reference<T2>::type>())) *);
    template <typename T2> static std::false_type test(...);
    static const bool value = std::is_same<std::true_type, decltype(test<T>(nullptr))>::value;
};

/// Helper type to replace 'void' in some expressions
struct void_type { };

/// to_string variant which also accepts strings
template <typename T> inline typename std::enable_if<!std::is_enum<T>::value, std::string>::type
to_string(const T &value) { return std::to_string(value); }
template <> inline std::string to_string(const std::string &value) { return value; }
template <typename T> inline typename std::enable_if<std::is_enum<T>::value, std::string>::type
to_string(T value) { return std::to_string((int) value); }

NAMESPACE_END(detail)

// C++ bindings of core Python exceptions
struct stop_iteration    : public std::runtime_error { public: stop_iteration(const std::string &w="") : std::runtime_error(w)   {} };
struct index_error       : public std::runtime_error { public: index_error(const std::string &w="")    : std::runtime_error(w)   {} };
struct error_already_set : public std::runtime_error { public: error_already_set() : std::runtime_error(detail::error_string())  {} };
/// Thrown when pybind11::cast or handle::call fail due to a type casting error
struct cast_error        : public std::runtime_error { public: cast_error(const std::string &w = "") : std::runtime_error(w)     {} };

PYBIND11_NOINLINE inline void pybind11_fail(const char *reason) { throw std::runtime_error(reason); }
PYBIND11_NOINLINE inline void pybind11_fail(const std::string &reason) { throw std::runtime_error(reason); }

NAMESPACE_END(pybind11)
