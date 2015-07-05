/*
    pybind/common.h -- Basic macros

    Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#if !defined(__PYBIND_COMMON_H)
#define __PYBIND_COMMON_H

#if !defined(NAMESPACE_BEGIN)
#define NAMESPACE_BEGIN(name) namespace name {
#endif
#if !defined(NAMESPACE_END)
#define NAMESPACE_END(name) }
#endif

#if !defined(PYTHON_EXPORT)
#if defined(WIN32)
#define PYTHON_EXPORT __declspec(dllexport)
#else
#define PYTHON_EXPORT __attribute__ ((visibility("default")))
#endif
#endif

#define PYTHON_PLUGIN(name) \
    extern "C" PYTHON_EXPORT PyObject *PyInit_##name()

#include <vector>
#include <string>
#include <stdexcept>
#include <functional>
#include <unordered_map>
#include <iostream>
#include <memory>

/// Include Python header, disable linking to pythonX_d.lib on Windows in debug mode
#if defined(_MSC_VER)
#define HAVE_ROUND
#pragma warning(push)
#pragma warning(disable: 4510 4610 4512)
#if _DEBUG
#define _DEBUG_MARKER
#undef _DEBUG
#endif
#endif
#include <Python.h>
#if defined(_MSC_VER)
#if defined(_DEBUG_MARKER)
#define _DEBUG
#undef _DEBUG_MARKER
#endif
#pragma warning(pop)
#endif

NAMESPACE_BEGIN(pybind)

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
template<> struct format_descriptor<int8_t>   { static std::string value() { return "b"; }; };
template<> struct format_descriptor<uint8_t>  { static std::string value() { return "B"; }; };
template<> struct format_descriptor<int16_t>  { static std::string value() { return "h"; }; };
template<> struct format_descriptor<uint16_t> { static std::string value() { return "H"; }; };
template<> struct format_descriptor<int32_t>  { static std::string value() { return "i"; }; };
template<> struct format_descriptor<uint32_t> { static std::string value() { return "I"; }; };
template<> struct format_descriptor<int64_t>  { static std::string value() { return "q"; }; };
template<> struct format_descriptor<uint64_t> { static std::string value() { return "Q"; }; };
template<> struct format_descriptor<float>    { static std::string value() { return "f"; }; };
template<> struct format_descriptor<double>   { static std::string value() { return "d"; }; };

/// Information record describing a Python buffer object
struct buffer_info {
    void *ptr;
    size_t itemsize;
    std::string format; // for dense contents, this should be set to format_descriptor<T>::value
    int ndim;
    std::vector<size_t> shape;
    std::vector<size_t> strides;

    buffer_info(void *ptr, size_t itemsize, const std::string &format,
                int ndim, const std::vector<size_t> &shape,
                const std::vector<size_t> &strides)
        : ptr(ptr), itemsize(itemsize), format(format), ndim(ndim),
          shape(shape), strides(strides) {}
};

// C++ bindings of core Python exceptions
struct stop_iteration    : public std::runtime_error { public: stop_iteration(const std::string &w="") : std::runtime_error(w)   {} };
struct index_error       : public std::runtime_error { public: index_error(const std::string &w="")    : std::runtime_error(w)   {} };
struct error_already_set : public std::exception     { public: error_already_set()                                               {} };
/// Thrown when pybind::cast or handle::call fail due to a type casting error
struct cast_error        : public std::runtime_error { public: cast_error(const std::string &w = "") : std::runtime_error(w) {} };

NAMESPACE_BEGIN(detail)

/// PyObject wrapper around generic types
template <typename type, typename holder_type = std::unique_ptr<type>> struct instance {
    PyObject_HEAD
    type *value;
    PyObject *parent;
    bool owned : 1;
    bool constructed : 1;
    holder_type holder;
};

/// Additional type information which does not fit into the PyTypeObjet
struct type_info {
    PyTypeObject *type;
    size_t type_size;
    void (*init_holder)(PyObject *);
    std::function<buffer_info *(PyObject *)> get_buffer;
    std::vector<PyObject *(*)(PyObject *, PyTypeObject *)> implicit_conversions;
};

/// Internal data struture used to track registered instances and types 
struct internals {
    std::unordered_map<std::string, type_info> registered_types;
    std::unordered_map<void *, PyObject *> registered_instances;
};

inline internals &get_internals();

NAMESPACE_END(detail)
NAMESPACE_END(pybind)

#endif /* __PYBIND_COMMON_H */
