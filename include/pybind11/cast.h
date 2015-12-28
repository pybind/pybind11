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
#include <array>
#include <limits>

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

#if PY_MAJOR_VERSION >= 3
#define PYBIND11_AS_STRING PyBytes_AsString
#else
#define PYBIND11_AS_STRING PyString_AsString
#endif

/** Linked list descriptor type for function signatures (produces smaller binaries
    compared to a previous solution using std::string and operator +=) */
class descr {
public:
    struct entry {
        const std::type_info *type = nullptr;
        const char *str = nullptr;
        entry *next = nullptr;
        entry(const std::type_info *type) : type(type) { }
        entry(const char *str) : str(str) { }
    };

    descr() { }
    descr(descr &&d) : first(d.first), last(d.last) { d.first = d.last = nullptr; }
    PYBIND11_NOINLINE descr(const char *str) { first = last = new entry { str }; }
    PYBIND11_NOINLINE descr(const std::type_info &type) { first = last = new entry { &type }; }

    PYBIND11_NOINLINE void operator+(const char *str) {
        entry *next = new entry { str };
        last->next = next;
        last = next;
    }

    PYBIND11_NOINLINE void operator+(const std::type_info *type) {
        entry *next = new entry { type };
        last->next = next;
        last = next;
    }

    PYBIND11_NOINLINE void operator+=(descr &&other) {
        last->next = other.first;
        while (last->next)
            last = last->next;
        other.first = other.last = nullptr;
    }

    PYBIND11_NOINLINE friend descr operator+(descr &&l, descr &&r) {
        descr result(std::move(l));
        result += std::move(r);
        return result;
    }

    PYBIND11_NOINLINE std::string str() const {
        std::string result;
        auto const& registered_types = get_internals().registered_types;
        for (entry *it = first; it != nullptr; it = it->next) {
            if (it->type) {
                auto it2 = registered_types.find(it->type);
                if (it2 != registered_types.end()) {
                    result += it2->second.type->tp_name;
                } else {
                    std::string tname(it->type->name());
                    detail::clean_type_id(tname);
                    result += tname;
                }
            } else {
                result += it->str;
            }
        }
        return result;
    }

    PYBIND11_NOINLINE ~descr() {
        while (first) {
            entry *tmp = first->next;
            delete first;
            first = tmp;
        }
    }

    entry *first = nullptr;
    entry *last = nullptr;
};

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
    static descr name() { return typeid(type); }

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
        static descr name() { return py_name; } \
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
                py_value = (py_type) detail::PyLong_AsLongLong_(src);
            else
                py_value = (py_type) detail::PyLong_AsUnsignedLongLong_(src);
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

    static descr name() {
        return std::is_floating_point<T>::value ? "float" : "int";
    }

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
    PYBIND11_TYPE_CASTER(void_type, "None");
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
    PYBIND11_TYPE_CASTER(bool, "bool");
};

template <> class type_caster<std::string> {
public:
    bool load(PyObject *src, bool) {
#if PY_MAJOR_VERSION < 3
        if (PyString_Check(src)) { value = PyString_AsString(src); return true; }
#endif
        object temp(PyUnicode_AsUTF8String(src), false);
        const char *ptr = nullptr;
        if (temp)
            ptr = PYBIND11_AS_STRING(temp.ptr());
        if (!ptr) { PyErr_Clear(); return false; }
        value = ptr;
        return true;
    }
    static PyObject *cast(const std::string &src, return_value_policy /* policy */, PyObject * /* parent */) {
        return PyUnicode_FromString(src.c_str());
    }
    PYBIND11_TYPE_CASTER(std::string, "str");
};

template <> class type_caster<char> {
public:
    bool load(PyObject *src, bool) {
#if PY_MAJOR_VERSION < 3
        if (PyString_Check(src)) { value = PyString_AsString(src); return true; }
#endif
        object temp(PyUnicode_AsUTF8String(src), false);
        const char *ptr = nullptr;
        if (temp)
            ptr = PYBIND11_AS_STRING(temp.ptr());
        if (!ptr) { PyErr_Clear(); return false; }
        value = ptr;
        return true;
    }

    static PyObject *cast(const char *src, return_value_policy /* policy */, PyObject * /* parent */) {
        return PyUnicode_FromString(src);
    }

    static PyObject *cast(char src, return_value_policy /* policy */, PyObject * /* parent */) {
        char str[2] = { src, '\0' };
        return PyUnicode_DecodeLatin1(str, 1, nullptr);
    }

    static descr name() { return "str"; }

    operator char*() { return (char *) value.c_str(); }
    operator char() { if (value.length() > 0) return value[0]; else return '\0'; }
protected:
    std::string value;
};

template <typename T1, typename T2> class type_caster<std::pair<T1, T2>> {
    typedef std::pair<T1, T2> type;
public:
    bool load(PyObject *src, bool convert) {
        if (!PyTuple_Check(src) || PyTuple_Size(src) != 2)
            return false;
        if (!first.load(PyTuple_GetItem(src, 0), convert))
            return false;
        return second.load(PyTuple_GetItem(src, 1), convert);
    }

    static PyObject *cast(const type &src, return_value_policy policy, PyObject *parent) {
        PyObject *o1 = type_caster<typename decay<T1>::type>::cast(src.first, policy, parent);
        PyObject *o2 = type_caster<typename decay<T2>::type>::cast(src.second, policy, parent);
        if (!o1 || !o2) {
            Py_XDECREF(o1);
            Py_XDECREF(o2);
            return nullptr;
        }
        PyObject *tuple = PyTuple_New(2);
        PyTuple_SetItem(tuple, 0, o1);
        PyTuple_SetItem(tuple, 1, o2);
        return tuple;
    }

    static descr name() {
        class descr result("(");
        result += std::move(type_caster<typename decay<T1>::type>::name());
        result += ", ";
        result += std::move(type_caster<typename decay<T2>::type>::name());
        result += ")";
        return result;
    }

    operator type() {
        return type(first, second);
    }
protected:
    type_caster<typename decay<T1>::type> first;
    type_caster<typename decay<T2>::type> second;
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

    static descr name(const char **keywords = nullptr, const char **values = nullptr) {
        std::array<class descr, size> names {{
            type_caster<typename decay<Tuple>::type>::name()...
        }};
        class descr result("(");
        for (int i=0; i<size; ++i) {
            if (keywords && keywords[i]) {
                result += keywords[i];
                result += " : ";
            }
            result += std::move(names[i]);
            if (values && values[i]) {
                result += " = ";
                result += values[i];
            }
            if (i+1 < size)
                result += ", ";
        }
        result += ")";
        return result;
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
        std::array<PyObject *, size> results {{
            type_caster<typename decay<Tuple>::type>::cast(std::get<Indices>(src), policy, parent)...
        }};
        bool success = true;
        for (auto result : results)
            if (result == nullptr)
                success = false;
        if (success) {
            PyObject *tuple = PyTuple_New(size);
            int counter = 0;
            for (auto result : results)
                PyTuple_SetItem(tuple, counter++, result);
            return tuple;
        } else {
            for (auto result : results) {
                Py_XDECREF(result);
            }
            return nullptr;
        }
    }

protected:
    std::tuple<type_caster<typename decay<Tuple>::type>...> value;
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

template <typename type>
struct type_caster<type, typename std::enable_if<std::is_base_of<handle, type>::value>::type> {
public:
    template <typename T = type, typename std::enable_if<std::is_same<T, handle>::value, int>::type = 0>
    bool load(PyObject *src, bool /* convert */) { value = handle(src); return value.check(); }

    template <typename T = type, typename std::enable_if<!std::is_same<T, handle>::value, int>::type = 0>
    bool load(PyObject *src, bool /* convert */) { value = type(src, true); return value.check(); }

    static PyObject *cast(const handle &src, return_value_policy /* policy */, PyObject * /* parent */) {
        src.inc_ref();
        return (PyObject *) src.ptr();
    }
    PYBIND11_TYPE_CASTER(type, typeid(type));
};

NAMESPACE_END(detail)

template <typename T> inline T cast(PyObject *object) {
    detail::type_caster<typename detail::decay<T>::type> conv;
    if (!conv.load(object, true))
        throw cast_error("Unable to cast Python object to C++ type");
    return conv;
}

template <typename T> inline object cast(const T &value, return_value_policy policy = return_value_policy::automatic, PyObject *parent = nullptr) {
    if (policy == return_value_policy::automatic)
        policy = std::is_pointer<T>::value ? return_value_policy::take_ownership : return_value_policy::copy;
    return object(detail::type_caster<typename detail::decay<T>::type>::cast(value, policy, parent), false);
}

template <typename T> inline T handle::cast() const { return pybind11::cast<T>(m_ptr); }
template <> inline void handle::cast() const { return; }

template <typename... Args> inline object handle::call(Args&&... args_) {
    const size_t size = sizeof...(Args);
    std::array<PyObject *, size> args{
        { detail::type_caster<typename detail::decay<Args>::type>::cast(
            std::forward<Args>(args_), return_value_policy::reference, nullptr)... }
    };
    bool fail = false;
    for (auto result : args)
        if (result == nullptr)
            fail = true;
    if (fail) {
        for (auto result : args) {
            Py_XDECREF(result);
        }
        throw cast_error("handle::call(): unable to convert input arguments to Python objects");
    }
    PyObject *tuple = PyTuple_New(size);
    int counter = 0;
    for (auto result : args)
        PyTuple_SetItem(tuple, counter++, result);
    PyObject *result = PyObject_CallObject(m_ptr, tuple);
    Py_DECREF(tuple);
    if (result == nullptr && PyErr_Occurred())
        throw error_already_set();
    return object(result, false);
}

NAMESPACE_END(pybind11)
