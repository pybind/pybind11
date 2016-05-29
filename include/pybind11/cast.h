/*
    pybind11/cast.h: Partial template specializations to cast between
    C++ and Python types

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "pytypes.h"
#include "typeid.h"
#include "descr.h"
#include <array>
#include <limits>
#include <iostream>

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

/// Additional type information which does not fit into the PyTypeObject
struct type_info {
    PyTypeObject *type;
    size_t type_size;
    void (*init_holder)(PyObject *, const void *);
    std::vector<PyObject *(*)(PyObject *, PyTypeObject *) > implicit_conversions;
    buffer_info *(*get_buffer)(PyObject *, void *) = nullptr;
    void *get_buffer_data = nullptr;
};

PYBIND11_NOINLINE inline internals &get_internals() {
    static internals *internals_ptr = nullptr;
    if (internals_ptr)
        return *internals_ptr;
    handle builtins(PyEval_GetBuiltins());
    const char *id = PYBIND11_INTERNALS_ID;
    capsule caps(builtins[id]);
    if (caps.check()) {
        internals_ptr = caps;
    } else {
        internals_ptr = new internals();
        #if defined(WITH_THREAD)
            PyEval_InitThreads();
            PyThreadState *tstate = PyThreadState_Get();
            internals_ptr->tstate = PyThread_create_key();
            PyThread_set_key_value(internals_ptr->tstate, tstate);
            internals_ptr->istate = tstate->interp;
        #endif
        builtins[id] = capsule(internals_ptr);
    }
    return *internals_ptr;
}

PYBIND11_NOINLINE inline detail::type_info* get_type_info(PyTypeObject *type, bool throw_if_missing = true) {
    auto const &type_dict = get_internals().registered_types_py;
    do {
        auto it = type_dict.find(type);
        if (it != type_dict.end())
            return (detail::type_info *) it->second;
        type = type->tp_base;
        if (!type) {
            if (throw_if_missing)
                pybind11_fail("pybind11::detail::get_type_info: unable to find type object!");
            return nullptr;
        }
    } while (true);
}

PYBIND11_NOINLINE inline detail::type_info *get_type_info(const std::type_info &tp) {
    auto &types = get_internals().registered_types_cpp;

    auto it = types.find(std::type_index(tp));
    if (it != types.end())
        return (detail::type_info *) it->second;
    return nullptr;
}

PYBIND11_NOINLINE inline handle get_type_handle(const std::type_info &tp) {
    detail::type_info *type_info = get_type_info(tp);
    return handle(type_info ? ((PyObject *) type_info->type) : nullptr);
}

PYBIND11_NOINLINE inline std::string error_string() {
    std::string errorString;
    PyThreadState *tstate = PyThreadState_GET();
    if (tstate == nullptr)
        return "";

    if (tstate->curexc_type) {
        errorString += (std::string) handle(tstate->curexc_type).str();
        errorString += ": ";
    }
    if (tstate->curexc_value)
        errorString += (std::string) handle(tstate->curexc_value).str();

    return errorString;
}

PYBIND11_NOINLINE inline handle get_object_handle(const void *ptr) {
    auto instances = get_internals().registered_instances;
    auto it = instances.find(ptr);
    if (it == instances.end())
        return handle();
    return handle((PyObject *) it->second);
}

inline PyThreadState *get_thread_state_unchecked() {
#if   PY_VERSION_HEX < 0x03000000
    return _PyThreadState_Current;
#elif PY_VERSION_HEX < 0x03050000
    return (PyThreadState*) _Py_atomic_load_relaxed(&_PyThreadState_Current);
#elif PY_VERSION_HEX < 0x03050200
    return (PyThreadState*) _PyThreadState_Current.value;
#else
    return _PyThreadState_UncheckedGet();
#endif
}

class type_caster_generic {
public:
    PYBIND11_NOINLINE type_caster_generic(const std::type_info &type_info)
     : typeinfo(get_type_info(type_info)) { }

    PYBIND11_NOINLINE bool load(handle src, bool convert) {
        if (!src || !typeinfo)
            return false;
        if (src.ptr() == Py_None) {
            value = nullptr;
            return true;
        } else if (PyType_IsSubtype(Py_TYPE(src.ptr()), typeinfo->type)) {
            value = ((instance<void> *) src.ptr())->value;
            return true;
        }
        if (convert) {
            for (auto &converter : typeinfo->implicit_conversions) {
                temp = object(converter(src.ptr(), typeinfo->type), false);
                if (load(temp, false))
                    return true;
            }
        }
        return false;
    }

    PYBIND11_NOINLINE static handle cast(const void *_src, return_value_policy policy, handle parent,
                                         const std::type_info *type_info,
                                         const std::type_info *type_info_backup,
                                         void *(*copy_constructor)(const void *),
                                         void *(*move_constructor)(const void *),
                                         const void *existing_holder = nullptr) {
        void *src = const_cast<void *>(_src);
        if (src == nullptr)
            return handle(Py_None).inc_ref();

        // avoid an issue with internal references matching their parent's address
        bool dont_cache = policy == return_value_policy::reference_internal &&
                          parent && ((instance<void> *) parent.ptr())->value == (void *) src;

        auto& internals = get_internals();
        auto it_instance = internals.registered_instances.find(src);
        if (it_instance != internals.registered_instances.end() && !dont_cache)
            return handle((PyObject *) it_instance->second).inc_ref();

        auto it = internals.registered_types_cpp.find(std::type_index(*type_info));
        if (it == internals.registered_types_cpp.end()) {
            type_info = type_info_backup;
            it = internals.registered_types_cpp.find(std::type_index(*type_info));
        }

        if (it == internals.registered_types_cpp.end()) {
            std::string tname = type_info->name();
            detail::clean_type_id(tname);
            std::string msg = "Unregistered type : " + tname;
            PyErr_SetString(PyExc_TypeError, msg.c_str());
            return handle();
        }

        auto tinfo = (const detail::type_info *) it->second;
        object inst(PyType_GenericAlloc(tinfo->type, 0), false);

        auto wrapper = (instance<void> *) inst.ptr();

        wrapper->value = src;
        wrapper->owned = true;
        wrapper->parent = nullptr;

        if (policy == return_value_policy::automatic)
            policy = return_value_policy::take_ownership;
        else if (policy == return_value_policy::automatic_reference)
            policy = return_value_policy::reference;

        if (policy == return_value_policy::copy) {
            wrapper->value = copy_constructor(wrapper->value);
            if (wrapper->value == nullptr)
                throw cast_error("return_value_policy = copy, but the object is non-copyable!");
        } else if (policy == return_value_policy::move) {
            wrapper->value = move_constructor(wrapper->value);
            if (wrapper->value == nullptr)
                wrapper->value = copy_constructor(wrapper->value);
            if (wrapper->value == nullptr)
                throw cast_error("return_value_policy = move, but the object is neither movable nor copyable!");
        } else if (policy == return_value_policy::reference) {
            wrapper->owned = false;
        } else if (policy == return_value_policy::reference_internal) {
            wrapper->owned = false;
            wrapper->parent = parent.inc_ref().ptr();
        }

        tinfo->init_holder(inst.ptr(), existing_holder);
        if (!dont_cache)
            internals.registered_instances[wrapper->value] = inst.ptr();

        return inst.release();
    }

protected:
    const type_info *typeinfo = nullptr;
    void *value = nullptr;
    object temp;
};

/* Determine suitable casting operator */
template <typename T>
using cast_op_type = typename std::conditional<std::is_pointer<typename std::remove_reference<T>::type>::value,
    typename std::add_pointer<typename intrinsic_type<T>::type>::type,
    typename std::add_lvalue_reference<typename intrinsic_type<T>::type>::type>::type;

/// Generic type caster for objects stored on the heap
template <typename type> class type_caster_base : public type_caster_generic {
public:
    static PYBIND11_DESCR name() { return type_descr(_<type>()); }

    type_caster_base() : type_caster_generic(typeid(type)) { }

    static handle cast(const type &src, return_value_policy policy, handle parent) {
        if (policy == return_value_policy::automatic || policy == return_value_policy::automatic_reference)
            policy = return_value_policy::copy;
        return cast(&src, policy, parent);
    }

    static handle cast(type &&src, return_value_policy policy, handle parent) {
        if (policy == return_value_policy::automatic || policy == return_value_policy::automatic_reference)
            policy = return_value_policy::move;
        return cast(&src, policy, parent);
    }

    static handle cast(const type *src, return_value_policy policy, handle parent) {
        return type_caster_generic::cast(
            src, policy, parent, src ? &typeid(*src) : nullptr, &typeid(type),
            make_copy_constructor(src), make_move_constructor(src));
    }

    template <typename T> using cast_op_type = pybind11::detail::cast_op_type<T>;

    operator type*() { return (type *) value; }
    operator type&() { if (!value) throw cast_error(); return *((type *) value); }

protected:
    typedef void *(*Constructor)(const void *stream);
#if !defined(_MSC_VER)
    /* Only enabled when the types are {copy,move}-constructible *and* when the type
       does not have a private operator new implementaton. */
    template <typename T = type> static auto make_copy_constructor(const T *value) -> decltype(new T(*value), Constructor(nullptr)) {
        return [](const void *arg) -> void * { return new T(*((const T *) arg)); }; }
    template <typename T = type> static auto make_move_constructor(const T *value) -> decltype(new T(std::move(*((T *) value))), Constructor(nullptr)) {
        return [](const void *arg) -> void * { return (void *) new T(std::move(*((T *) arg))); }; }
#else
    /* Visual Studio 2015's SFINAE implementation doesn't yet handle the above robustly in all situations.
       Use a workaround that only tests for constructibility for now. */
    template <typename T = type, typename = typename std::enable_if<std::is_copy_constructible<T>::value>::type>
    static Constructor make_copy_constructor(const T *value) {
        return [](const void *arg) -> void * { return new T(*((const T *)arg)); }; }
    template <typename T = type, typename = typename std::enable_if<std::is_move_constructible<T>::value>::type>
    static Constructor make_move_constructor(const T *value) {
        return [](const void *arg) -> void * { return (void *) new T(std::move(*((T *)arg))); }; }
#endif
    static Constructor make_copy_constructor(...) { return nullptr; }
    static Constructor make_move_constructor(...) { return nullptr; }
};

template <typename type, typename SFINAE = void> class type_caster : public type_caster_base<type> { };

template <typename type> class type_caster<std::reference_wrapper<type>> : public type_caster_base<type> {
public:
    static handle cast(const std::reference_wrapper<type> &src, return_value_policy policy, handle parent) {
        return type_caster_base<type>::cast(&src.get(), policy, parent);
    }
    template <typename T> using cast_op_type = std::reference_wrapper<type>;
    operator std::reference_wrapper<type>() { return std::ref(*((type *) this->value)); }
};

#define PYBIND11_TYPE_CASTER(type, py_name) \
    protected: \
        type value; \
    public: \
        static PYBIND11_DESCR name() { return type_descr(py_name); } \
        static handle cast(const type *src, return_value_policy policy, handle parent) { \
            return cast(*src, policy, parent); \
        } \
        operator type*() { return &value; } \
        operator type&() { return value; } \
        template <typename _T> using cast_op_type = pybind11::detail::cast_op_type<_T>

#define PYBIND11_DECLARE_HOLDER_TYPE(type, holder_type) \
    namespace pybind11 { namespace detail { \
    template <typename type> class type_caster<holder_type> \
        : public type_caster_holder<type, holder_type> { }; \
    }}


template <typename T>
struct type_caster<
    T, typename std::enable_if<std::is_integral<T>::value ||
                               std::is_floating_point<T>::value>::type> {
    typedef typename std::conditional<sizeof(T) <= sizeof(long), long, long long>::type _py_type_0;
    typedef typename std::conditional<std::is_signed<T>::value, _py_type_0, typename std::make_unsigned<_py_type_0>::type>::type _py_type_1;
    typedef typename std::conditional<std::is_floating_point<T>::value, double, _py_type_1>::type py_type;
public:

    bool load(handle src, bool) {
        py_type py_value;

        if (!src) {
            return false;
        } if (std::is_floating_point<T>::value) {
            py_value = (py_type) PyFloat_AsDouble(src.ptr());
        } else if (sizeof(T) <= sizeof(long)) {
            if (PyFloat_Check(src.ptr()))
                return false;
            if (std::is_signed<T>::value)
                py_value = (py_type) PyLong_AsLong(src.ptr());
            else
                py_value = (py_type) PyLong_AsUnsignedLong(src.ptr());
        } else {
            if (PyFloat_Check(src.ptr()))
                return false;
            if (std::is_signed<T>::value)
                py_value = (py_type) PYBIND11_LONG_AS_LONGLONG(src.ptr());
            else
                py_value = (py_type) PYBIND11_LONG_AS_UNSIGNED_LONGLONG(src.ptr());
        }

        if ((py_value == (py_type) -1 && PyErr_Occurred()) ||
            (std::is_integral<T>::value && sizeof(py_type) != sizeof(T) &&
               (py_value < (py_type) std::numeric_limits<T>::min() ||
                py_value > (py_type) std::numeric_limits<T>::max()))) {
            PyErr_Clear();
            return false;
        }

        value = (T) py_value;
        return true;
    }

    static handle cast(T src, return_value_policy /* policy */, handle /* parent */) {
        if (std::is_floating_point<T>::value) {
            return PyFloat_FromDouble((double) src);
        } else if (sizeof(T) <= sizeof(long)) {
            if (std::is_signed<T>::value)
                return PyLong_FromLong((long) src);
            else
                return PyLong_FromUnsignedLong((unsigned long) src);
        } else {
            if (std::is_signed<T>::value)
                return PyLong_FromLongLong((long long) src);
            else
                return PyLong_FromUnsignedLongLong((unsigned long long) src);
        }
    }

    static handle cast(const T *src, return_value_policy policy, handle parent) {
        return cast(*src, policy, parent);
    }

    template <typename T2 = T, typename std::enable_if<std::is_integral<T2>::value, int>::type = 0>
    static PYBIND11_DESCR name() { return type_descr(_("int")); }

    template <typename T2 = T, typename std::enable_if<!std::is_integral<T2>::value, int>::type = 0>
    static PYBIND11_DESCR name() { return type_descr(_("float")); }

    operator T*() { return &value; }
    operator T&() { return value; }

    template <typename T2> using cast_op_type = pybind11::detail::cast_op_type<T2>;
protected:
    T value;
};

template <> class type_caster<void_type> {
public:
    bool load(handle, bool) { return false; }
    static handle cast(void_type, return_value_policy /* policy */, handle /* parent */) {
        return handle(Py_None).inc_ref();
    }
    PYBIND11_TYPE_CASTER(void_type, _("NoneType"));
};

template <> class type_caster<void> : public type_caster<void_type> {
public:
    using type_caster<void_type>::cast;

    bool load(handle h, bool) {
        if (!h) {
            return false;
        } else if (h.ptr() == Py_None) {
            value = nullptr;
            return true;
        }

        /* Check if this is a capsule */
        capsule c(h, true);
        if (c.check()) {
            value = (void *) c;
            return true;
        }

        /* Check if this is a C++ type */
        if (get_type_info((PyTypeObject *) h.get_type().ptr(), false)) {
            value = ((instance<void> *) h.ptr())->value;
            return true;
        }

        /* Fail */
        return false;
    }

    static handle cast(const void *ptr, return_value_policy /* policy */, handle /* parent */) {
        if (ptr)
            return capsule(ptr).release();
        else
            return handle(Py_None).inc_ref();
    }

    template <typename T> using cast_op_type = void*&;
    operator void *&() { return value; }
    static PYBIND11_DESCR name() { return _("capsule"); }
private:
    void *value = nullptr;
};

template <> class type_caster<std::nullptr_t> : public type_caster<void_type> { };

template <> class type_caster<bool> {
public:
    bool load(handle src, bool) {
        if (!src) return false;
        else if (src.ptr() == Py_True) { value = true; return true; }
        else if (src.ptr() == Py_False) { value = false; return true; }
        else return false;
    }
    static handle cast(bool src, return_value_policy /* policy */, handle /* parent */) {
        return handle(src ? Py_True : Py_False).inc_ref();
    }
    PYBIND11_TYPE_CASTER(bool, _("bool"));
};

template <> class type_caster<std::string> {
public:
    bool load(handle src, bool) {
        object temp;
        handle load_src = src;
        if (!src) {
            return false;
        } else if (PyUnicode_Check(load_src.ptr())) {
            temp = object(PyUnicode_AsUTF8String(load_src.ptr()), false);
            if (!temp) { PyErr_Clear(); return false; }  // UnicodeEncodeError
            load_src = temp;
        }
        char *buffer;
        ssize_t length;
        int err = PYBIND11_BYTES_AS_STRING_AND_SIZE(load_src.ptr(), &buffer, &length);
        if (err == -1) { PyErr_Clear(); return false; }  // TypeError
        value = std::string(buffer, (size_t) length);
        success = true;
        return true;
    }

    static handle cast(const std::string &src, return_value_policy /* policy */, handle /* parent */) {
        return PyUnicode_FromStringAndSize(src.c_str(), (ssize_t) src.length());
    }

    PYBIND11_TYPE_CASTER(std::string, _(PYBIND11_STRING_NAME));
protected:
    bool success = false;
};

template <typename type> class type_caster<std::unique_ptr<type>> {
public:
    static handle cast(std::unique_ptr<type> &&src, return_value_policy policy, handle parent) {
        handle result = type_caster_base<type>::cast(src.get(), policy, parent);
        if (result)
            src.release();
        return result;
    }
    static PYBIND11_DESCR name() { return type_caster_base<type>::name(); }
};

template <> class type_caster<std::wstring> {
public:
    bool load(handle src, bool) {
        object temp;
        handle load_src = src;
        if (!src) {
            return false;
        } else if (!PyUnicode_Check(load_src.ptr())) {
            temp = object(PyUnicode_FromObject(load_src.ptr()), false);
            if (!temp) { PyErr_Clear(); return false; }
            load_src = temp;
        }
        wchar_t *buffer = nullptr;
        ssize_t length = -1;
#if PY_MAJOR_VERSION >= 3
        buffer = PyUnicode_AsWideCharString(load_src.ptr(), &length);
#else
        temp = object(
            sizeof(wchar_t) == sizeof(short)
                ? PyUnicode_AsUTF16String(load_src.ptr())
                : PyUnicode_AsUTF32String(load_src.ptr()), false);
        if (temp) {
            int err = PYBIND11_BYTES_AS_STRING_AND_SIZE(temp.ptr(), (char **) &buffer, &length);
            if (err == -1) { buffer = nullptr; }  // TypeError
            length = length / (ssize_t) sizeof(wchar_t) - 1; ++buffer; // Skip BOM
        }
#endif
        if (!buffer) { PyErr_Clear(); return false; }
        value = std::wstring(buffer, (size_t) length);
        success = true;
        return true;
    }

    static handle cast(const std::wstring &src, return_value_policy /* policy */, handle /* parent */) {
        return PyUnicode_FromWideChar(src.c_str(), (ssize_t) src.length());
    }

    PYBIND11_TYPE_CASTER(std::wstring, _(PYBIND11_STRING_NAME));
protected:
    bool success = false;
};

template <> class type_caster<char> : public type_caster<std::string> {
public:
    bool load(handle src, bool convert) {
        if (src.ptr() == Py_None) return true;
        return type_caster<std::string>::load(src, convert);
    }

    static handle cast(const char *src, return_value_policy /* policy */, handle /* parent */) {
        if (src == nullptr) return handle(Py_None).inc_ref();
        return PyUnicode_FromString(src);
    }

    static handle cast(char src, return_value_policy /* policy */, handle /* parent */) {
        char str[2] = { src, '\0' };
        return PyUnicode_DecodeLatin1(str, 1, nullptr);
    }

    operator char*() { return success ? (char *) value.c_str() : nullptr; }
    operator char&() { return value[0]; }

    static PYBIND11_DESCR name() { return type_descr(_(PYBIND11_STRING_NAME)); }
};

template <> class type_caster<wchar_t> : public type_caster<std::wstring> {
public:
    bool load(handle src, bool convert) {
        if (src.ptr() == Py_None) return true;
        return type_caster<std::wstring>::load(src, convert);
    }

    static handle cast(const wchar_t *src, return_value_policy /* policy */, handle /* parent */) {
        if (src == nullptr) return handle(Py_None).inc_ref();
        return PyUnicode_FromWideChar(src, (ssize_t) wcslen(src));
    }

    static handle cast(wchar_t src, return_value_policy /* policy */, handle /* parent */) {
        wchar_t wstr[2] = { src, L'\0' };
        return PyUnicode_FromWideChar(wstr, 1);
    }

    operator wchar_t*() { return success ? (wchar_t *) value.c_str() : nullptr; }
    operator wchar_t&() { return value[0]; }

    static PYBIND11_DESCR name() { return type_descr(_(PYBIND11_STRING_NAME)); }
};

template <typename T1, typename T2> class type_caster<std::pair<T1, T2>> {
    typedef std::pair<T1, T2> type;
public:
    bool load(handle src, bool convert) {
        if (!src)
            return false;
        else if (!PyTuple_Check(src.ptr()) || PyTuple_Size(src.ptr()) != 2)
            return false;
        return  first.load(PyTuple_GET_ITEM(src.ptr(), 0), convert) &&
               second.load(PyTuple_GET_ITEM(src.ptr(), 1), convert);
    }

    static handle cast(const type &src, return_value_policy policy, handle parent) {
        object o1 = object(type_caster<typename intrinsic_type<T1>::type>::cast(src.first, policy, parent), false);
        object o2 = object(type_caster<typename intrinsic_type<T2>::type>::cast(src.second, policy, parent), false);
        if (!o1 || !o2)
            return handle();
        tuple result(2);
        PyTuple_SET_ITEM(result.ptr(), 0, o1.release().ptr());
        PyTuple_SET_ITEM(result.ptr(), 1, o2.release().ptr());
        return result.release();
    }

    static PYBIND11_DESCR name() {
        return type_descr(
            _("(") + type_caster<typename intrinsic_type<T1>::type>::name() +
            _(", ") + type_caster<typename intrinsic_type<T2>::type>::name() + _(")"));
    }

    template <typename T> using cast_op_type = type;

    operator type() {
        return type(first .operator typename type_caster<typename intrinsic_type<T1>::type>::template cast_op_type<T1>(),
                    second.operator typename type_caster<typename intrinsic_type<T2>::type>::template cast_op_type<T2>());
    }
protected:
    type_caster<typename intrinsic_type<T1>::type> first;
    type_caster<typename intrinsic_type<T2>::type> second;
};

template <typename... Tuple> class type_caster<std::tuple<Tuple...>> {
    typedef std::tuple<Tuple...> type;
    typedef std::tuple<typename intrinsic_type<Tuple>::type...> itype;
    typedef std::tuple<args> args_type;
    typedef std::tuple<args, kwargs> args_kwargs_type;
public:
    enum { size = sizeof...(Tuple) };

    static constexpr const bool has_kwargs = std::is_same<itype, args_kwargs_type>::value;
    static constexpr const bool has_args = has_kwargs || std::is_same<itype, args_type>::value;

    bool load(handle src, bool convert) {
        if (!src || !PyTuple_Check(src.ptr()) || PyTuple_GET_SIZE(src.ptr()) != size)
            return false;
        return load(src, convert, typename make_index_sequence<sizeof...(Tuple)>::type());
    }

    template <typename T = itype, typename std::enable_if<
        !std::is_same<T, args_type>::value &&
        !std::is_same<T, args_kwargs_type>::value, int>::type = 0>
    bool load_args(handle args, handle, bool convert) {
        return load(args, convert, typename make_index_sequence<sizeof...(Tuple)>::type());
    }

    template <typename T = itype, typename std::enable_if<std::is_same<T, args_type>::value, int>::type = 0>
    bool load_args(handle args, handle, bool convert) {
        std::get<0>(value).load(args, convert);
        return true;
    }

    template <typename T = itype, typename std::enable_if<std::is_same<T, args_kwargs_type>::value, int>::type = 0>
    bool load_args(handle args, handle kwargs, bool convert) {
        std::get<0>(value).load(args, convert);
        std::get<1>(value).load(kwargs, convert);
        return true;
    }

    static handle cast(const type &src, return_value_policy policy, handle parent) {
        return cast(src, policy, parent, typename make_index_sequence<size>::type());
    }

    static PYBIND11_DESCR name() {
        return type_descr(
               _("(") +
               detail::concat(type_caster<typename intrinsic_type<Tuple>::type>::name()...) +
               _(")"));
    }

    template <typename ReturnValue, typename Func> typename std::enable_if<!std::is_void<ReturnValue>::value, ReturnValue>::type call(Func &&f) {
        return call<ReturnValue>(std::forward<Func>(f), typename make_index_sequence<sizeof...(Tuple)>::type());
    }

    template <typename ReturnValue, typename Func> typename std::enable_if<std::is_void<ReturnValue>::value, void_type>::type call(Func &&f) {
        call<ReturnValue>(std::forward<Func>(f), typename make_index_sequence<sizeof...(Tuple)>::type());
        return void_type();
    }

    template <typename T> using cast_op_type = type;

    operator type() {
        return cast(typename make_index_sequence<sizeof...(Tuple)>::type());
    }

protected:
    template <typename ReturnValue, typename Func, size_t ... Index> ReturnValue call(Func &&f, index_sequence<Index...>) {
        return f(std::get<Index>(value)
            .operator typename type_caster<typename intrinsic_type<Tuple>::type>::template cast_op_type<Tuple>()...);
    }

    template <size_t ... Index> type cast(index_sequence<Index...>) {
        return type(std::get<Index>(value)
            .operator typename type_caster<typename intrinsic_type<Tuple>::type>::template cast_op_type<Tuple>()...);
    }

    template <size_t ... Indices> bool load(handle src, bool convert, index_sequence<Indices...>) {
        std::array<bool, size> success {{
            std::get<Indices>(value).load(PyTuple_GET_ITEM(src.ptr(), Indices), convert)...
        }};
        (void) convert; /* avoid a warning when the tuple is empty */
        for (bool r : success)
            if (!r)
                return false;
        return true;
    }

    /* Implementation: Convert a C++ tuple into a Python tuple */
    template <size_t ... Indices> static handle cast(const type &src, return_value_policy policy, handle parent, index_sequence<Indices...>) {
        std::array<object, size> entries {{
            object(type_caster<typename intrinsic_type<Tuple>::type>::cast(std::get<Indices>(src), policy, parent), false)...
        }};
        for (const auto &entry: entries)
            if (!entry)
                return handle();
        tuple result(size);
        int counter = 0;
        for (auto & entry: entries)
            PyTuple_SET_ITEM(result.ptr(), counter++, entry.release().ptr());
        return result.release();
    }

protected:
    std::tuple<type_caster<typename intrinsic_type<Tuple>::type>...> value;
};

/// Type caster for holder types like std::shared_ptr, etc.
template <typename type, typename holder_type> class type_caster_holder : public type_caster_base<type> {
public:
    using type_caster_base<type>::cast;
    using type_caster_base<type>::typeinfo;
    using type_caster_base<type>::value;
    using type_caster_base<type>::temp;

    bool load(handle src, bool convert) {
        if (!src || !typeinfo) {
            return false;
        } else if (src.ptr() == Py_None) {
            value = nullptr;
            return true;
        } else if (PyType_IsSubtype(Py_TYPE(src.ptr()), typeinfo->type)) {
            auto inst = (instance<type, holder_type> *) src.ptr();
            value = (void *) inst->value;
            holder = inst->holder;
            return true;
        }

        if (convert) {
            for (auto &converter : typeinfo->implicit_conversions) {
                temp = object(converter(src.ptr(), typeinfo->type), false);
                if (load(temp, false))
                    return true;
            }
        }
        return false;
    }

    explicit operator type*() { return this->value; }
    explicit operator type&() { return *(this->value); }
    explicit operator holder_type*() { return &holder; }

    // Workaround for Intel compiler bug
    // see pybind11 issue 94
    #if defined(__ICC) || defined(__INTEL_COMPILER)
    operator holder_type&() { return holder; }
    #else
    explicit operator holder_type&() { return holder; }
    #endif

    static handle cast(const holder_type &src, return_value_policy, handle) {
        return type_caster_generic::cast(
            src.get(), return_value_policy::take_ownership, handle(),
            src.get() ? &typeid(*src.get()) : nullptr, &typeid(type),
            nullptr, nullptr, &src);
    }

protected:
    holder_type holder;
};

template <typename T> struct handle_type_name { static PYBIND11_DESCR name() { return _<T>(); } };
template <> struct handle_type_name<bytes> { static PYBIND11_DESCR name() { return _(PYBIND11_BYTES_NAME); } };
template <> struct handle_type_name<args> { static PYBIND11_DESCR name() { return _("*args"); } };
template <> struct handle_type_name<kwargs> { static PYBIND11_DESCR name() { return _("**kwargs"); } };

template <typename type>
struct type_caster<type, typename std::enable_if<std::is_base_of<handle, type>::value>::type> {
public:
    template <typename T = type, typename std::enable_if<!std::is_base_of<object, T>::value, int>::type = 0>
    bool load(handle src, bool /* convert */) { value = type(src); return value.check(); }

    template <typename T = type, typename std::enable_if<std::is_base_of<object, T>::value, int>::type = 0>
    bool load(handle src, bool /* convert */) { value = type(src, true); return value.check(); }

    static handle cast(const handle &src, return_value_policy /* policy */, handle /* parent */) {
        return src.inc_ref();
    }
    PYBIND11_TYPE_CASTER(type, handle_type_name<type>::name());
};

NAMESPACE_END(detail)

template <typename T> T cast(handle handle) {
    typedef detail::type_caster<typename detail::intrinsic_type<T>::type> type_caster;
    type_caster conv;
    if (!conv.load(handle, true))
        throw cast_error("Unable to cast Python object to C++ type");
    return conv.operator typename type_caster::template cast_op_type<T>();
}

template <typename T> object cast(const T &value,
        return_value_policy policy = return_value_policy::automatic_reference,
        handle parent = handle()) {
    if (policy == return_value_policy::automatic)
        policy = std::is_pointer<T>::value ? return_value_policy::take_ownership : return_value_policy::copy;
    else if (policy == return_value_policy::automatic_reference)
        policy = std::is_pointer<T>::value ? return_value_policy::reference : return_value_policy::copy;
    return object(detail::type_caster<typename detail::intrinsic_type<T>::type>::cast(value, policy, parent), false);
}

template <typename T> T handle::cast() const { return pybind11::cast<T>(*this); }
template <> inline void handle::cast() const { return; }

template <return_value_policy policy = return_value_policy::automatic_reference,
          typename... Args> tuple make_tuple(Args&&... args_) {
    const size_t size = sizeof...(Args);
    std::array<object, size> args {
        { object(detail::type_caster<typename detail::intrinsic_type<Args>::type>::cast(
            std::forward<Args>(args_), policy, nullptr), false)... }
    };
    for (auto &arg_value : args)
        if (!arg_value)
            throw cast_error("make_tuple(): unable to convert arguments to Python objects");
    tuple result(size);
    int counter = 0;
    for (auto &arg_value : args)
        PyTuple_SET_ITEM(result.ptr(), counter++, arg_value.release().ptr());
    return result;
}

template <typename... Args> object handle::operator()(Args&&... args) const {
    tuple args_tuple = pybind11::make_tuple(std::forward<Args>(args)...);
    object result(PyObject_CallObject(m_ptr, args_tuple.ptr()), false);
    if (!result)
        throw error_already_set();
    return result;
}

template <typename... Args> object handle::call(Args &&... args) const {
    return operator()(std::forward<Args>(args)...);
}

inline object handle::operator()(detail::args_proxy args) const {
    object result(PyObject_CallObject(m_ptr, args.ptr()), false);
    if (!result)
        throw error_already_set();
    return result;
}

inline object handle::operator()(detail::args_proxy args, detail::kwargs_proxy kwargs) const {
    object result(PyObject_Call(m_ptr, args.ptr(), kwargs.ptr()), false);
    if (!result)
        throw error_already_set();
    return result;
}

#define PYBIND11_MAKE_OPAQUE(Type) \
    namespace pybind11 { namespace detail { \
        template<> class type_caster<Type> : public type_caster_base<Type> { }; \
    }}

NAMESPACE_END(pybind11)
