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

class type_caster_custom {
public:
    PYBIND11_NOINLINE type_caster_custom(const std::type_info *type_info) {
        auto & registered_types = get_internals().registered_types;
        auto it = registered_types.find(type_info);
        if (it != registered_types.end()) {
            typeinfo = &it->second;
        } else {
            /* Unknown type?! Since std::type_info* often varies across
               module boundaries, the following does an explicit check */
            for (auto const &type : registered_types) {
                if (strcmp(type.first->name(), type_info->name()) == 0) {
                    registered_types[type_info] = type.second;
                    typeinfo = &type.second;
                    break;
                }
            }
        }
    }

    PYBIND11_NOINLINE bool load(PyObject *src, bool convert) {
        if (src == nullptr || typeinfo == nullptr)
            return false;
        if (PyType_IsSubtype(Py_TYPE(src), typeinfo->type)) {
            value = ((instance<void> *) src)->value;
            return true;
        }
        if (convert) {
            for (auto &converter : typeinfo->implicit_conversions) {
                temp = object(converter(src, typeinfo->type), false);
                if (load(temp.ptr(), false))
                    return true;
            }
        }
        return false;
    }

    PYBIND11_NOINLINE static PyObject *cast(const void *_src, return_value_policy policy, PyObject *parent,
                                            const std::type_info *type_info, void *(*copy_constructor)(const void *)) {
        void *src = const_cast<void *>(_src);
        if (src == nullptr) {
            Py_INCREF(Py_None);
            return Py_None;
        }
        // avoid an issue with internal references matching their parent's address
        bool dont_cache = policy == return_value_policy::reference_internal &&
                          parent && ((instance<void> *) parent)->value == (void *) src;
        auto& internals = get_internals();
        auto it_instance = internals.registered_instances.find(src);
        if (it_instance != internals.registered_instances.end() && !dont_cache) {
            PyObject *inst = it_instance->second;
            Py_INCREF(inst);
            return inst;
        }
        auto it = internals.registered_types.find(type_info);
        if (it == internals.registered_types.end()) {
            std::string tname = type_info->name();
            detail::clean_type_id(tname);
            std::string msg = "Unregistered type : " + tname;
            PyErr_SetString(PyExc_TypeError, msg.c_str());
            return nullptr;
        }
        auto &reg_type = it->second;
        instance<void> *inst = (instance<void> *) PyType_GenericAlloc(reg_type.type, 0);
        inst->value = src;
        inst->owned = true;
        inst->parent = nullptr;
        if (policy == return_value_policy::automatic)
            policy = return_value_policy::take_ownership;
        if (policy == return_value_policy::copy) {
            inst->value = copy_constructor(inst->value);
            if (inst->value == nullptr)
                throw cast_error("return_value_policy = copy, but the object is non-copyable!");
        } else if (policy == return_value_policy::reference) {
            inst->owned = false;
        } else if (policy == return_value_policy::reference_internal) {
            inst->owned = false;
            inst->parent = parent;
            Py_XINCREF(parent);
        }
        PyObject *inst_pyobj = (PyObject *) inst;
        reg_type.init_holder(inst_pyobj);
        if (!dont_cache)
            internals.registered_instances[inst->value] = inst_pyobj;
        return inst_pyobj;
    }

protected:
    const type_info *typeinfo = nullptr;
    void *value = nullptr;
    object temp;
};

/// Generic type caster for objects stored on the heap
template <typename type, typename Enable = void> class type_caster : public type_caster_custom {
public:
    static PYBIND11_DESCR name() { return type_descr(_<type>()); }

    type_caster() : type_caster_custom(&typeid(type)) { }

    static PyObject *cast(const type &src, return_value_policy policy, PyObject *parent) {
        if (policy == return_value_policy::automatic)
            policy = return_value_policy::copy;
        return type_caster_custom::cast(&src, policy, parent, &typeid(type), &copy_constructor);
    }

    static PyObject *cast(const type *src, return_value_policy policy, PyObject *parent) {
        return type_caster_custom::cast(src, policy, parent, &typeid(type), &copy_constructor);
    }

    operator type*() { return (type *) value; }
    operator type&() { return *((type *) value); }
protected:
    template <typename T = type, typename std::enable_if<std::is_copy_constructible<T>::value, int>::type = 0>
    static void *copy_constructor(const void *arg) {
        return new type(*((const type *)arg));
    }
    template <typename T = type, typename std::enable_if<!std::is_copy_constructible<T>::value, int>::type = 0>
    static void *copy_constructor(const void *) { return nullptr; }
};

#define PYBIND11_TYPE_CASTER(type, py_name) \
    protected: \
        type value; \
    public: \
        static PYBIND11_DESCR name() { return type_descr(py_name); } \
        static PyObject *cast(const type *src, return_value_policy policy, PyObject *parent) { \
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

    bool load(PyObject *src, bool) {
        py_type py_value;

        if (std::is_floating_point<T>::value) {
            py_value = (py_type) PyFloat_AsDouble(src);
        } else if (sizeof(T) <= sizeof(long)) {
            if (std::is_signed<T>::value)
                py_value = (py_type) PyLong_AsLong(src);
            else
                py_value = (py_type) PyLong_AsUnsignedLong(src);
        } else {
            if (std::is_signed<T>::value)
                py_value = (py_type) PYBIND11_LONG_AS_LONGLONG(src);
            else
                py_value = (py_type) PYBIND11_LONG_AS_UNSIGNED_LONGLONG(src);
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

    static PyObject *cast(T src, return_value_policy /* policy */, PyObject * /* parent */) {
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

    static PyObject *cast(const T *src, return_value_policy policy, PyObject *parent) {
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
    bool load(PyObject *, bool) { return false; }
    static PyObject *cast(void_type, return_value_policy /* policy */, PyObject * /* parent */) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    PYBIND11_TYPE_CASTER(void_type, _("NoneType"));
};

template <> class type_caster<void> : public type_caster<void_type> { };
template <> class type_caster<std::nullptr_t> : public type_caster<void_type> { };

template <> class type_caster<bool> {
public:
    bool load(PyObject *src, bool) {
        if (src == Py_True) { value = true; return true; }
        else if (src == Py_False) { value = false; return true; }
        else return false;
    }
    static PyObject *cast(bool src, return_value_policy /* policy */, PyObject * /* parent */) {
        PyObject *result = src ? Py_True : Py_False;
        Py_INCREF(result);
        return result;
    }
    PYBIND11_TYPE_CASTER(bool, _("bool"));
};

template <> class type_caster<std::string> {
public:
    bool load(PyObject *src, bool) {
        object temp;
        PyObject *load_src = src;
        if (PyUnicode_Check(src)) {
            temp = object(PyUnicode_AsUTF8String(src), false);
            if (!temp) { PyErr_Clear(); return false; }  // UnicodeEncodeError
            load_src = temp.ptr();
        }
        char *buffer;
        ssize_t length;
        int err = PYBIND11_BYTES_AS_STRING_AND_SIZE(load_src, &buffer, &length);
        if (err == -1) { PyErr_Clear(); return false; }  // TypeError
        value = std::string(buffer, length);
        return true;
    }

    static PyObject *cast(const std::string &src, return_value_policy /* policy */, PyObject * /* parent */) {
        return PyUnicode_FromStringAndSize(src.c_str(), src.length());
    }

    PYBIND11_TYPE_CASTER(std::string, _(PYBIND11_STRING_NAME));
};

template <> class type_caster<char> {
public:
    bool load(PyObject *src, bool) {
        object temp;
        PyObject *load_src = src;
        if (PyUnicode_Check(src)) {
            temp = object(PyUnicode_AsUTF8String(src), false);
            if (!temp) { PyErr_Clear(); return false; }  // UnicodeEncodeError
            load_src = temp.ptr();
        }
        const char *ptr = PYBIND11_BYTES_AS_STRING(load_src);
        if (!ptr) { PyErr_Clear(); return false; }  // TypeError
        value = std::string(ptr);
        return true;
    }

    static PyObject *cast(const char *src, return_value_policy /* policy */, PyObject * /* parent */) {
        return PyUnicode_FromString(src);
    }

    static PyObject *cast(char src, return_value_policy /* policy */, PyObject * /* parent */) {
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
    bool load(PyObject *src, bool convert) {
        if (!PyTuple_Check(src) || PyTuple_Size(src) != 2)
            return false;
        if (!first.load(PyTuple_GET_ITEM(src, 0), convert))
            return false;
        return second.load(PyTuple_GET_ITEM(src, 1), convert);
    }

    static PyObject *cast(const type &src, return_value_policy policy, PyObject *parent) {
        object o1(type_caster<typename intrinsic_type<T1>::type>::cast(src.first, policy, parent), false);
        object o2(type_caster<typename intrinsic_type<T2>::type>::cast(src.second, policy, parent), false);
        if (!o1 || !o2)
            return nullptr;
        PyObject *tuple = PyTuple_New(2);
        if (!tuple)
            return nullptr;
        PyTuple_SET_ITEM(tuple, 0, o1.release());
        PyTuple_SET_ITEM(tuple, 1, o2.release());
        return tuple;
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

    bool load(PyObject *src, bool convert) {
        return load(src, convert, typename make_index_sequence<sizeof...(Tuple)>::type());
    }

    static PyObject *cast(const type &src, return_value_policy policy, PyObject *parent) {
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

    template <size_t ... Indices> bool load(PyObject *src, bool convert, index_sequence<Indices...>) {
        if (!PyTuple_Check(src))
            return false;
        if (PyTuple_Size(src) != size)
            return false;
        std::array<bool, size> results {{
            (PyTuple_GET_ITEM(src, Indices) != nullptr ? std::get<Indices>(value).load(PyTuple_GET_ITEM(src, Indices), convert) : false)...
        }};
        (void) convert; /* avoid a warning when the tuple is empty */
        for (bool r : results)
            if (!r)
                return false;
        return true;
    }

    /* Implementation: Convert a C++ tuple into a Python tuple */
    template <size_t ... Indices> static PyObject *cast(const type &src, return_value_policy policy, PyObject *parent, index_sequence<Indices...>) {
        std::array<object, size> results {{
            object(type_caster<typename intrinsic_type<Tuple>::type>::cast(std::get<Indices>(src), policy, parent), false)...
        }};
        for (const auto & result : results)
            if (!result)
                return nullptr;
        PyObject *tuple = PyTuple_New(size);
        if (!tuple)
            return nullptr;
        int counter = 0;
        for (auto & result : results)
            PyTuple_SET_ITEM(tuple, counter++, result.release());
        return tuple;
    }

protected:
    std::tuple<type_caster<typename intrinsic_type<Tuple>::type>...> value;
};

/// Type caster for holder types like std::shared_ptr, etc.
template <typename type, typename holder_type> class type_caster_holder : public type_caster<type> {
public:
    typedef type_caster<type> parent;

    template <typename T = holder_type,
              typename std::enable_if<std::is_same<std::shared_ptr<type>, T>::value, int>::type = 0>
    bool load(PyObject *src, bool convert) {
        if (!parent::load(src, convert))
            return false;
        holder = holder_type(((type *) parent::value)->shared_from_this());
        return true;
    }

    template <typename T = holder_type,
              typename std::enable_if<!std::is_same<std::shared_ptr<type>, T>::value, int>::type = 0>
    bool load(PyObject *src, bool convert) {
        if (!parent::load(src, convert))
            return false;
        holder = holder_type((type *) parent::value);
        return true;
    }

    explicit operator type*() { return this->value; }
    explicit operator type&() { return *(this->value); }
    explicit operator holder_type&() { return holder; }
    explicit operator holder_type*() { return &holder; }

    using type_caster<type>::cast;
    static PyObject *cast(const holder_type &src, return_value_policy policy, PyObject *parent) {
        return type_caster<type>::cast(src.get(), policy, parent);
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
    bool load(PyObject *src, bool /* convert */) { value = handle(src); return value.check(); }

    template <typename T = type, typename std::enable_if<!std::is_same<T, handle>::value, int>::type = 0>
    bool load(PyObject *src, bool /* convert */) { value = type(src, true); return value.check(); }

    static PyObject *cast(const handle &src, return_value_policy /* policy */, PyObject * /* parent */) {
        src.inc_ref(); return (PyObject *) src.ptr();
    }
    PYBIND11_TYPE_CASTER(type, handle_type_name<type>::name());
};

NAMESPACE_END(detail)

template <typename T> inline T cast(PyObject *object) {
    detail::type_caster<typename detail::intrinsic_type<T>::type> conv;
    if (!conv.load(object, true))
        throw cast_error("Unable to cast Python object to C++ type");
    return conv;
}

template <typename T> inline object cast(const T &value, return_value_policy policy = return_value_policy::automatic, PyObject *parent = nullptr) {
    if (policy == return_value_policy::automatic)
        policy = std::is_pointer<T>::value ? return_value_policy::take_ownership : return_value_policy::copy;
    return object(detail::type_caster<typename detail::intrinsic_type<T>::type>::cast(value, policy, parent), false);
}

template <typename T> inline T handle::cast() const { return pybind11::cast<T>(m_ptr); }
template <> inline void handle::cast() const { return; }

template <typename... Args> inline object handle::call(Args&&... args_) const {
    const size_t size = sizeof...(Args);
    std::array<object, size> args{
        { object(detail::type_caster<typename detail::intrinsic_type<Args>::type>::cast(
            std::forward<Args>(args_), return_value_policy::reference, nullptr), false)... }
    };
    for (const auto & result : args)
        if (!result)
            throw cast_error("handle::call(): unable to convert input arguments to Python objects");
    object tuple(PyTuple_New(size), false);
    if (!tuple)
        throw cast_error("handle::call(): unable to allocate tuple");
    int counter = 0;
    for (auto & result : args)
        PyTuple_SET_ITEM(tuple.ptr(), counter++, result.release());
    PyObject *result = PyObject_CallObject(m_ptr, tuple.ptr());
    if (result == nullptr && PyErr_Occurred())
        throw error_already_set();
    return object(result, false);
}

NAMESPACE_END(pybind11)
