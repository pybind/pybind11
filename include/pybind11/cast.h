/*
    pybind11/cast.h: Partial template specializations to cast between
    C++ and Python types

    Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "pytypes.h"
#include "typeid.h"
#include "descr.h"
#include <array>
#include <limits>

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
    capsule caps(builtins["__pybind11__"]);
    if (caps.check()) {
        internals_ptr = caps;
    } else {
        internals_ptr = new internals();
        builtins["__pybind11__"] = capsule(internals_ptr);
    }
    return *internals_ptr;
}

PYBIND11_NOINLINE inline detail::type_info* get_type_info(PyTypeObject *type) {
    auto const &type_dict = get_internals().registered_types_py;
    do {
        auto it = type_dict.find(type);
        if (it != type_dict.end())
            return (detail::type_info *) it->second;
        type = type->tp_base;
        if (!type)
            pybind11_fail("pybind11::detail::get_type_info: unable to find type object!");
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

class type_caster_generic {
public:
    PYBIND11_NOINLINE type_caster_generic(const std::type_info &type_info)
     : typeinfo(get_type_info(type_info)) { }

    PYBIND11_NOINLINE bool load(handle src, bool convert) {
        if (!src || !typeinfo)
            return false;
        if (PyType_IsSubtype(Py_TYPE(src.ptr()), typeinfo->type)) {
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
                                         void *(*copy_constructor)(const void *),
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

        if (policy == return_value_policy::copy) {
            wrapper->value = copy_constructor(wrapper->value);
            if (wrapper->value == nullptr)
                throw cast_error("return_value_policy = copy, but the object is non-copyable!");
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

/// Generic type caster for objects stored on the heap
template <typename type, typename Enable = void> class type_caster : public type_caster_generic {
public:
    static PYBIND11_DESCR name() { return type_descr(_<type>()); }

    type_caster() : type_caster_generic(typeid(type)) { }

    static handle cast(const type &src, return_value_policy policy, handle parent) {
        if (policy == return_value_policy::automatic)
            policy = return_value_policy::copy;
        return type_caster_generic::cast(&src, policy, parent, &typeid(type), &copy_constructor);
    }

    static handle cast(const type *src, return_value_policy policy, handle parent) {
        return type_caster_generic::cast(src, policy, parent, &typeid(type), &copy_constructor);
    }

    operator type*() { return (type *) value; }
    operator type&() { return *((type *) value); }
protected:
    template <typename T = type, typename std::enable_if<detail::is_copy_constructible<T>::value, int>::type = 0>
    static void *copy_constructor(const void *arg) {
        return new type(*((const type *) arg));
    }
    template <typename T = type, typename std::enable_if<!detail::is_copy_constructible<T>::value, int>::type = 0>
    static void *copy_constructor(const void *) { return nullptr; }
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
        operator type&() { return value; }

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

        if (std::is_floating_point<T>::value) {
            py_value = (py_type) PyFloat_AsDouble(src.ptr());
        } else if (sizeof(T) <= sizeof(long)) {
            if (std::is_signed<T>::value)
                py_value = (py_type) PyLong_AsLong(src.ptr());
            else
                py_value = (py_type) PyLong_AsUnsignedLong(src.ptr());
        } else {
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

template <> class type_caster<void> : public type_caster<void_type> { };
template <> class type_caster<std::nullptr_t> : public type_caster<void_type> { };

template <> class type_caster<bool> {
public:
    bool load(handle src, bool) {
        if (src.ptr() == Py_True) { value = true; return true; }
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
        if (PyUnicode_Check(load_src.ptr())) {
            temp = object(PyUnicode_AsUTF8String(load_src.ptr()), false);
            if (!temp) { PyErr_Clear(); return false; }  // UnicodeEncodeError
            load_src = temp;
        }
        char *buffer;
        ssize_t length;
        int err = PYBIND11_BYTES_AS_STRING_AND_SIZE(load_src.ptr(), &buffer, &length);
        if (err == -1) { PyErr_Clear(); return false; }  // TypeError
        value = std::string(buffer, length);
        return true;
    }

    static handle cast(const std::string &src, return_value_policy /* policy */, handle /* parent */) {
        return PyUnicode_FromStringAndSize(src.c_str(), src.length());
    }

    PYBIND11_TYPE_CASTER(std::string, _(PYBIND11_STRING_NAME));
};

template <> class type_caster<char> {
public:
    bool load(handle src, bool) {
        object temp;
        handle load_src = src;
        if (PyUnicode_Check(load_src.ptr())) {
            temp = object(PyUnicode_AsUTF8String(load_src.ptr()), false);
            if (!temp) { PyErr_Clear(); return false; }  // UnicodeEncodeError
            load_src = temp;
        }
        const char *ptr = PYBIND11_BYTES_AS_STRING(load_src.ptr());
        if (!ptr) { PyErr_Clear(); return false; }  // TypeError
        value = std::string(ptr);
        return true;
    }

    static handle cast(const char *src, return_value_policy /* policy */, handle /* parent */) {
        return PyUnicode_FromString(src);
    }

    static handle cast(char src, return_value_policy /* policy */, handle /* parent */) {
        char str[2] = { src, '\0' };
        return PyUnicode_DecodeLatin1(str, 1, nullptr);
    }

    operator char*() { return (char *) value.c_str(); }
    operator char() { if (value.length() > 0) return value[0]; else return '\0'; }

    static PYBIND11_DESCR name() { return type_descr(_(PYBIND11_STRING_NAME)); }
protected:
    std::string value;
};

template <typename T1, typename T2> class type_caster<std::pair<T1, T2>> {
    typedef std::pair<T1, T2> type;
public:
    bool load(handle src, bool convert) {
        if (!PyTuple_Check(src.ptr()) || PyTuple_Size(src.ptr()) != 2)
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

    operator type() {
        return type(first, second);
    }
protected:
    type_caster<typename intrinsic_type<T1>::type> first;
    type_caster<typename intrinsic_type<T2>::type> second;
};

template <typename... Tuple> class type_caster<std::tuple<Tuple...>> {
    typedef std::tuple<Tuple...> type;
public:
    enum { size = sizeof...(Tuple) };

    bool load(handle src, bool convert) {
        return load(src, convert, typename make_index_sequence<sizeof...(Tuple)>::type());
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

    operator type() {
        return cast(typename make_index_sequence<sizeof...(Tuple)>::type());
    }

protected:
    template <typename ReturnValue, typename Func, size_t ... Index> ReturnValue call(Func &&f, index_sequence<Index...>) {
        return f((Tuple) std::get<Index>(value)...);
    }

    template <size_t ... Index> type cast(index_sequence<Index...>) {
        return type((Tuple) std::get<Index>(value)...);
    }

    template <size_t ... Indices> bool load(handle src, bool convert, index_sequence<Indices...>) {
        if (!PyTuple_Check(src.ptr()) || PyTuple_Size(src.ptr()) != size)
            return false;
        std::array<bool, size> success {{
            (PyTuple_GET_ITEM(src.ptr(), Indices) != nullptr ? std::get<Indices>(value).load(PyTuple_GET_ITEM(src.ptr(), Indices), convert) : false)...
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
template <typename type, typename holder_type> class type_caster_holder : public type_caster<type> {
public:
    using type_caster<type>::cast;
    using type_caster<type>::typeinfo;
    using type_caster<type>::value;
    using type_caster<type>::temp;
    using type_caster<type>::copy_constructor;

    bool load(handle src, bool convert) {
        if (!src || !typeinfo)
            return false;

        if (PyType_IsSubtype(Py_TYPE(src.ptr()), typeinfo->type)) {
            auto inst = (instance<type, holder_type> *) src.ptr();
            value = inst->value;
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
    explicit operator holder_type&() { return holder; }
    explicit operator holder_type*() { return &holder; }

    static handle cast(const holder_type &src, return_value_policy policy, handle parent) {
        return type_caster_generic::cast(
            src.get(), policy, parent, &typeid(type), &copy_constructor, &src);
    }

protected:
    holder_type holder;
};

template <typename T> struct handle_type_name { static PYBIND11_DESCR name() { return _<T>(); } };
template <> struct handle_type_name<bytes> { static PYBIND11_DESCR name() { return _(PYBIND11_BYTES_NAME); } };

template <typename type>
struct type_caster<type, typename std::enable_if<std::is_base_of<handle, type>::value>::type> {
public:
    template <typename T = type, typename std::enable_if<std::is_same<T, handle>::value, int>::type = 0>
    bool load(handle src, bool /* convert */) { value = src; return value.check(); }

    template <typename T = type, typename std::enable_if<!std::is_same<T, handle>::value, int>::type = 0>
    bool load(handle src, bool /* convert */) { value = type(src, true); return value.check(); }

    static handle cast(const handle &src, return_value_policy /* policy */, handle /* parent */) {
        return src.inc_ref();
    }
    PYBIND11_TYPE_CASTER(type, handle_type_name<type>::name());
};

NAMESPACE_END(detail)

template <typename T> inline T cast(handle handle) {
    detail::type_caster<typename detail::intrinsic_type<T>::type> conv;
    if (!conv.load(handle, true))
        throw cast_error("Unable to cast Python object to C++ type");
    return conv;
}

template <typename T> inline object cast(const T &value, return_value_policy policy = return_value_policy::automatic, handle parent = handle()) {
    if (policy == return_value_policy::automatic)
        policy = std::is_pointer<T>::value ? return_value_policy::take_ownership : return_value_policy::copy;
    return object(detail::type_caster<typename detail::intrinsic_type<T>::type>::cast(value, policy, parent), false);
}

template <typename T> inline T handle::cast() const { return pybind11::cast<T>(m_ptr); }
template <> inline void handle::cast() const { return; }

template <typename... Args> inline object handle::call(Args&&... args_) const {
    const size_t size = sizeof...(Args);
    std::array<object, size> args {
        { object(detail::type_caster<typename detail::intrinsic_type<Args>::type>::cast(
            std::forward<Args>(args_), return_value_policy::reference, nullptr), false)... }
    };
    for (auto &arg_value : args)
        if (!arg_value)
            throw cast_error("handle::call(): unable to convert input "
                             "arguments to Python objects");
    tuple args_tuple(size);
    int counter = 0;
    for (auto &arg_value : args)
        PyTuple_SET_ITEM(args_tuple.ptr(), counter++, arg_value.release().ptr());
    object result(PyObject_CallObject(m_ptr, args_tuple.ptr()), false);
    if (!result)
        throw error_already_set();
    return result;
}

NAMESPACE_END(pybind11)
