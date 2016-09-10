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
    // Implicit conversions from other types to this type via declared Python constructor
    std::vector<PyObject *(*)(PyObject *, PyTypeObject *) > implicit_conversions_python;
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
        internals_ptr->registered_exception_translators.push_front(
            [](std::exception_ptr p) -> void {
                try {
                    if (p) std::rethrow_exception(p);
                } catch (const error_already_set &)      {                                                 return;
                } catch (const index_error &e)           { PyErr_SetString(PyExc_IndexError,    e.what()); return;
                } catch (const key_error &e)             { PyErr_SetString(PyExc_KeyError,      e.what()); return;
                } catch (const value_error &e)           { PyErr_SetString(PyExc_ValueError,    e.what()); return;
                } catch (const type_error &e)            { PyErr_SetString(PyExc_TypeError,     e.what()); return;
                } catch (const stop_iteration &e)        { PyErr_SetString(PyExc_StopIteration, e.what()); return;
                } catch (const std::bad_alloc &e)        { PyErr_SetString(PyExc_MemoryError,   e.what()); return;
                } catch (const std::domain_error &e)     { PyErr_SetString(PyExc_ValueError,    e.what()); return;
                } catch (const std::invalid_argument &e) { PyErr_SetString(PyExc_ValueError,    e.what()); return;
                } catch (const std::length_error &e)     { PyErr_SetString(PyExc_ValueError,    e.what()); return;
                } catch (const std::out_of_range &e)     { PyErr_SetString(PyExc_IndexError,    e.what()); return;
                } catch (const std::range_error &e)      { PyErr_SetString(PyExc_ValueError,    e.what()); return;
                } catch (const std::exception &e)        { PyErr_SetString(PyExc_RuntimeError,  e.what()); return;
                } catch (...) {
                    PyErr_SetString(PyExc_RuntimeError, "Caught an unknown exception!");
                    return;
                }
            }
        );
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
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown internal error occurred");
        return "Unknown internal error occurred";
    }

    error_scope scope; // Preserve error state

    std::string errorString;
    if (scope.type) {
        errorString += handle(scope.type).attr("__name__").cast<std::string>();
        errorString += ": ";
    }
    if (scope.value)
        errorString += (std::string) handle(scope.value).str();

    return errorString;
}

PYBIND11_NOINLINE inline handle get_object_handle(const void *ptr, const detail::type_info *type ) {
    auto &instances = get_internals().registered_instances;
    auto range = instances.equal_range(ptr);
    for (auto it = range.first; it != range.second; ++it) {
        auto instance_type = detail::get_type_info(Py_TYPE(it->second), false);
        if (instance_type && instance_type == type)
            return handle((PyObject *) it->second);
    }
    return handle();
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

struct function_record; // Forward declaration
struct function_call_internals {
    const function_record &rec;
    handle args, kwargs, &parent;
    const std::vector<std::type_index> &arg_types;
    PyObject *py_args_tuple;
    // Set to true by the tuple caster to signal arguments that loaded normally
    std::vector<bool> loaded;
    // Stores pointers to instances of the argument type that will be passed to functions in lieu of
    // the (presumably) failed regular argument loading
    std::vector<void *> implicit_pointers_cpp;
    // Tracks the indices of implicit_pointers_cpp that hold implicit conversion instances that need
    // destruction when this function_call_internals is destroyed.
    std::vector<bool> implicit_pointer_is_managed;

    function_call_internals(
            function_record &rec,
            tuple &args,
            PyObject *kwargs,
            handle &parent,
            PyObject *py_args_tuple,
            std::vector<std::type_index> &arg_types
            ) :
        rec(rec), args(args), kwargs(kwargs), parent(parent), arg_types(arg_types),  py_args_tuple(py_args_tuple),
        loaded(arg_types.size(), false),
        implicit_pointers_cpp(arg_types.size(), nullptr),
        implicit_pointer_is_managed(arg_types.size(), false)
    {}

    ~function_call_internals() {
        auto &imp_conv = get_internals().implicit_conversions_cpp;

        for (size_t i = 0; i < implicit_pointers_cpp.size(); i++) {
            if (implicit_pointers_cpp[i] && implicit_pointer_is_managed[i]) {
                imp_conv[arg_types[i]].destructor(implicit_pointers_cpp[i]);
            }
        }
    }
};


// Currently the only "alternative" here is implicit C++ conversion, but other alternatives could
// also be supported here in the future.
PYBIND11_NOINLINE inline bool loaded_or_alternatives(function_call_internals &fvars) {
    auto &imp_conv = get_internals().implicit_conversions_cpp;

    // Store the constructors we find; we only actually construct if we find a constructor for all
    // unloaded types:
    std::vector<std::pair<size_t, void *(*)(void *)>> constructors;
    for (size_t i = 0; i < fvars.arg_types.size(); i++) {
        std::type_index arg_type = fvars.arg_types[i];
        if (fvars.loaded[i]) continue;
        auto type = Py_TYPE(PyTuple_GET_ITEM(fvars.py_args_tuple, i));

        bool found = false;
        for (auto &conversion : imp_conv[arg_type].constructors) {
            if (PyType_IsSubtype(type, conversion.first)) {
                auto &constructor = conversion.second;
                if (constructor) {
                    constructors.emplace_back(i, constructor);
                    found = true;
                }
                // Allow a null constructor reference to abort conversion attempts
                break;
            }
        }


        // If the i'th argument didn't load normally and we couldn't find any way to convert the
        // given argument, we can't help with the load.
        if (!found) return false;
    }

    // Call any constructors that we found for implicit conversion
    try {
        for (auto &constr : constructors) {
            // Construct new instance:
            fvars.implicit_pointers_cpp[constr.first] = constr.second(((detail::instance<void> *) PyTuple_GET_ITEM(fvars.py_args_tuple, constr.first))->value);
            fvars.implicit_pointer_is_managed[constr.first] = true;
        }
    }
    catch (...) {
        // If one of the constructors raised an exception we abort; this will trigger a failure back
        // to the dispatcher, which then lets the function_call_internals be destroyed, which
        // handles the cleanup of instances we created.
        return false;
    }

    return true;
}

// Forward declaration
inline void keep_alive_impl(handle nurse, handle patient);
class type_caster_generic {
public:
    PYBIND11_NOINLINE type_caster_generic(const std::type_info &type_info)
     : typeinfo(get_type_info(type_info)) { }

    PYBIND11_NOINLINE bool load(handle src, bool convert) {
        if (!src || !typeinfo)
            return false;
        if (src.is_none()) {
            value = nullptr;
            return true;
        } else if (PyType_IsSubtype(Py_TYPE(src.ptr()), typeinfo->type)) {
            value = ((instance<void> *) src.ptr())->value;
            return true;
        }

        if (convert) {
            // Walk through the implicit python conversions looking for one that works.
            for (auto &converter : typeinfo->implicit_conversions_python) {
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
            return none().inc_ref();

        auto &internals = get_internals();

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

        auto it_instances = internals.registered_instances.equal_range(src);
        for (auto it_i = it_instances.first; it_i != it_instances.second; ++it_i) {
            auto instance_type = detail::get_type_info(Py_TYPE(it_i->second), false);
            if (instance_type && instance_type == tinfo)
                return handle((PyObject *) it_i->second).inc_ref();
        }

        object inst(PyType_GenericAlloc(tinfo->type, 0), false);

        auto wrapper = (instance<void> *) inst.ptr();

        wrapper->value = src;
        wrapper->owned = true;

        if (policy == return_value_policy::automatic)
            policy = return_value_policy::take_ownership;
        else if (policy == return_value_policy::automatic_reference)
            policy = return_value_policy::reference;

        if (policy == return_value_policy::copy) {
            if (copy_constructor)
                wrapper->value = copy_constructor(wrapper->value);
            else
                throw cast_error("return_value_policy = copy, but the object is non-copyable!");
        } else if (policy == return_value_policy::move) {
            if (move_constructor)
                wrapper->value = move_constructor(wrapper->value);
            else if (copy_constructor)
                wrapper->value = copy_constructor(wrapper->value);
            else
                throw cast_error("return_value_policy = move, but the object is neither movable nor copyable!");
        } else if (policy == return_value_policy::reference) {
            wrapper->owned = false;
        } else if (policy == return_value_policy::reference_internal) {
            wrapper->owned = false;
            detail::keep_alive_impl(inst, parent);
        }

        tinfo->init_holder(inst.ptr(), existing_holder);

        internals.registered_instances.emplace(wrapper->value, inst.ptr());

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
    typename std::add_pointer<intrinsic_t<T>>::type,
    typename std::add_lvalue_reference<intrinsic_t<T>>::type>::type;

/// Generic type caster for objects stored on the heap
template <typename type> class type_caster_base : public type_caster_generic {
    using itype = intrinsic_t<type>;
public:
    static PYBIND11_DESCR name() { return type_descr(_<type>()); }

    type_caster_base() : type_caster_generic(typeid(type)) { }

    static handle cast(const itype &src, return_value_policy policy, handle parent) {
        if (policy == return_value_policy::automatic || policy == return_value_policy::automatic_reference)
            policy = return_value_policy::copy;
        return cast(&src, policy, parent);
    }

    static handle cast(itype &&src, return_value_policy policy, handle parent) {
        if (policy == return_value_policy::automatic || policy == return_value_policy::automatic_reference)
            policy = return_value_policy::move;
        return cast(&src, policy, parent);
    }

    static handle cast(const itype *src, return_value_policy policy, handle parent) {
        return type_caster_generic::cast(
            src, policy, parent, src ? &typeid(*src) : nullptr, &typeid(type),
            make_copy_constructor(src), make_move_constructor(src));
    }

    template <typename T> using cast_op_type = pybind11::detail::cast_op_type<T>;

    operator itype*() { return (type *) value; }
    operator itype&() { if (!value) throw reference_cast_error(); return *((itype *) value); }

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
template <typename type> using make_caster = type_caster<intrinsic_t<type>>;

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
struct type_caster<T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
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

    // The first descr below encodes the actual type, but we provide a signature type of "int" or
    // "float" rather than its more varied cpp names (unsigned long, short, double, etc.)
    PYBIND11_TYPE_CASTER(T, _<T>("") + _<std::is_integral<T>::value>("int", "float"));
};

template <> class type_caster<void_type> {
public:
    bool load(handle, bool) { return false; }
    static handle cast(void_type, return_value_policy /* policy */, handle /* parent */) {
        return none().inc_ref();
    }
    PYBIND11_TYPE_CASTER(void_type, _("None"));
};

template <> class type_caster<void> : public type_caster<void_type> {
public:
    using type_caster<void_type>::cast;

    bool load(handle h, bool) {
        if (!h) {
            return false;
        } else if (h.is_none()) {
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
            return none().inc_ref();
    }

    template <typename T> using cast_op_type = void*&;
    operator void *&() { return value; }
    static PYBIND11_DESCR name() { return type_descr(_("capsule")); }
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
    PYBIND11_TYPE_CASTER(bool, _<bool>("bool"));
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

    PYBIND11_TYPE_CASTER(std::string, _<std::string>(PYBIND11_STRING_NAME));
protected:
    bool success = false;
};

template <typename type, typename deleter> class type_caster<std::unique_ptr<type, deleter>> {
public:
    static handle cast(std::unique_ptr<type, deleter> &&src, return_value_policy policy, handle parent) {
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

    PYBIND11_TYPE_CASTER(std::wstring, _<std::wstring>(PYBIND11_STRING_NAME));
protected:
    bool success = false;
};

template <> class type_caster<char> : public type_caster<std::string> {
public:
    bool load(handle src, bool convert) {
        if (src.is_none()) return true;
        return type_caster<std::string>::load(src, convert);
    }

    static handle cast(const char *src, return_value_policy /* policy */, handle /* parent */) {
        if (src == nullptr) return none().inc_ref();
        return PyUnicode_FromString(src);
    }

    static handle cast(char src, return_value_policy /* policy */, handle /* parent */) {
        char str[2] = { src, '\0' };
        return PyUnicode_DecodeLatin1(str, 1, nullptr);
    }

    operator char*() { return success ? (char *) value.c_str() : nullptr; }
    operator char&() { return value[0]; }

    static PYBIND11_DESCR name() { return type_descr(_<char>(PYBIND11_STRING_NAME)); }
};

template <> class type_caster<wchar_t> : public type_caster<std::wstring> {
public:
    bool load(handle src, bool convert) {
        if (src.is_none()) return true;
        return type_caster<std::wstring>::load(src, convert);
    }

    static handle cast(const wchar_t *src, return_value_policy /* policy */, handle /* parent */) {
        if (src == nullptr) return none().inc_ref();
        return PyUnicode_FromWideChar(src, (ssize_t) wcslen(src));
    }

    static handle cast(wchar_t src, return_value_policy /* policy */, handle /* parent */) {
        wchar_t wstr[2] = { src, L'\0' };
        return PyUnicode_FromWideChar(wstr, 1);
    }

    operator wchar_t*() { return success ? (wchar_t *) value.c_str() : nullptr; }
    operator wchar_t&() { return value[0]; }

    static PYBIND11_DESCR name() { return type_descr(_<wchar_t>(PYBIND11_STRING_NAME)); }
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
        object o1 = object(make_caster<T1>::cast(src.first, policy, parent), false);
        object o2 = object(make_caster<T2>::cast(src.second, policy, parent), false);
        if (!o1 || !o2)
            return handle();
        tuple result(2);
        PyTuple_SET_ITEM(result.ptr(), 0, o1.release().ptr());
        PyTuple_SET_ITEM(result.ptr(), 1, o2.release().ptr());
        return result.release();
    }

    static PYBIND11_DESCR name() {
        return type_descr(
            _("Tuple[") + make_caster<T1>::name() + _(", ") + make_caster<T2>::name() + _("]")
        );
    }

    template <typename T> using cast_op_type = type;

    operator type() {
        return type(first.operator typename make_caster<T1>::template cast_op_type<T1>(),
                    second.operator typename make_caster<T2>::template cast_op_type<T2>());
    }
protected:
    make_caster<T1> first;
    make_caster<T2> second;
};

template <typename... Tuple> class type_caster<std::tuple<Tuple...>> {
    typedef std::tuple<Tuple...> type;
    typedef std::tuple<intrinsic_t<Tuple>...> itype;
    typedef std::tuple<args> args_type;
    typedef std::tuple<args, kwargs> args_kwargs_type;
protected:
    std::tuple<type_caster<typename intrinsic_type<Tuple>::type>...> value;
    // Function call data; must be set for any of the function call-related member functions
    function_call_internals *fvars;
public:
    static constexpr const bool has_kwargs = std::is_same<itype, args_kwargs_type>::value;
    static constexpr const bool has_args = has_kwargs || std::is_same<itype, args_type>::value;

    type_caster(function_call_internals *fvars = nullptr) : fvars{fvars} {}

    enum { size = sizeof...(Tuple) };
    bool load(handle src, bool convert) {
        if (!src || !PyTuple_Check(src.ptr()) || PyTuple_GET_SIZE(src.ptr()) != size)
            return false;
        return load(src, convert, typename make_index_sequence<size>::type());
    }
    static handle cast(const type &src, return_value_policy policy, handle parent) {
        return cast(src, policy, parent, typename make_index_sequence<size>::type());
    }
    template <typename T> using cast_op_type = type;
    operator type() {
        return cast(typename make_index_sequence<size>::type());
    }

    template <typename T = itype, typename std::enable_if<
        !std::is_same<T, args_type>::value &&
        !std::is_same<T, args_kwargs_type>::value, int>::type = 0>
    void load_args() {
        load_items(typename make_index_sequence<size>::type());
    }
    template <typename T = itype, typename std::enable_if<std::is_same<T, args_type>::value, int>::type = 0>
    void load_args() {
        std::get<0>(value).load(fvars->args, true);
        fvars->loaded[0] = true;
    }
    template <typename T = itype, typename std::enable_if<std::is_same<T, args_kwargs_type>::value, int>::type = 0>
    void load_args() {
        std::get<0>(value).load(fvars->args, true);
        std::get<1>(value).load(fvars->kwargs, true);
        fvars->loaded[0] = true;
        fvars->loaded[1] = true;
    }

    static PYBIND11_DESCR element_names() {
        return detail::concat(_(PYBIND11_DESCR_ARG_PREFIX) + make_caster<Tuple>::name()...);
    }

    static PYBIND11_DESCR name() {
        return type_descr(_("Tuple[") + element_names() + _("]"));
    }

    template <typename ReturnValue, typename Func> typename std::enable_if<!std::is_void<ReturnValue>::value, ReturnValue>::type call(Func &&f) {
        return call<ReturnValue>(std::forward<Func>(f), typename make_index_sequence<size>::type());
    }
    template <typename ReturnValue, typename Func> typename std::enable_if<std::is_void<ReturnValue>::value, void_type>::type call(Func &&f) {
        call<ReturnValue>(std::forward<Func>(f), typename make_index_sequence<size>::type());
        return void_type();
    }
protected:
    template <size_t ... Index> type cast(index_sequence<Index...>) {
        return type(std::get<Index>(value)
            .operator typename make_caster<Tuple>::template cast_op_type<Tuple>()...);
    }
    template <size_t ... Indices>
    bool load(handle src, bool convert, index_sequence<Indices...>) {
        std::array<bool, size> loaded = {{
            std::get<Indices>(value).load(PyTuple_GET_ITEM(src.ptr(), Indices), convert)...
        }};
        (void) convert; /* avoid a warning when the tuple is empty */
        for (bool r : loaded)
            if (!r)
                return false;
        return true;
    }

    template <size_t ... Indices>
    void load_items(index_sequence<Indices...>) {
        fvars->loaded = {
            std::get<Indices>(value).load(PyTuple_GET_ITEM(fvars->py_args_tuple, Indices), true)...
        };
    }

    /* Implementation: Convert a C++ tuple into a Python tuple */
    template <size_t ... Indices> static handle cast(const type &src, return_value_policy policy, handle parent, index_sequence<Indices...>) {
        std::array<object, size> entries {{
            object(make_caster<Tuple>::cast(std::get<Indices>(src), policy, parent), false)...
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

    template <typename ReturnValue, typename Func, size_t ... Index>
    ReturnValue call(Func &&f, index_sequence<Index...>) {
        return f(
            (fvars->implicit_pointers_cpp[Index]
                ? *reinterpret_cast<typename std::remove_reference<typename make_caster<Tuple>::template cast_op_type<Tuple>>::type*>(fvars->implicit_pointers_cpp[Index])
                : std::get<Index>(value).operator typename make_caster<Tuple>::template cast_op_type<Tuple>()
            )...);
    }
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
        } else if (src.is_none()) {
            value = nullptr;
            return true;
        } else if (PyType_IsSubtype(Py_TYPE(src.ptr()), typeinfo->type)) {
            auto inst = (instance<type, holder_type> *) src.ptr();
            value = (void *) inst->value;
            holder = inst->holder;
            return true;
        }

        if (convert) {
            // Walk through the implicit python conversions looking for one that works.
            for (auto &converter : typeinfo->implicit_conversions_python) {
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

// PYBIND11_DECLARE_HOLDER_TYPE holder types:
template <typename base, typename holder> struct is_holder_type :
    std::is_base_of<detail::type_caster_holder<base, holder>, detail::type_caster<holder>> {};
// Specialization for always-supported unique_ptr holders:
template <typename base, typename deleter> struct is_holder_type<base, std::unique_ptr<base, deleter>> :
    std::true_type {};

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

// Our conditions for enabling moving are quite restrictive:
// At compile time:
// - T needs to be a non-const, non-pointer, non-reference type
// - type_caster<T>::operator T&() must exist
// - the type must be move constructible (obviously)
// At run-time:
// - if the type is non-copy-constructible, the object must be the sole owner of the type (i.e. it
//   must have ref_count() == 1)h
// If any of the above are not satisfied, we fall back to copying.
template <typename T, typename SFINAE = void> struct move_is_plain_type : std::false_type {};
template <typename T> struct move_is_plain_type<T, typename std::enable_if<
        !std::is_void<T>::value && !std::is_pointer<T>::value && !std::is_reference<T>::value && !std::is_const<T>::value
    >::type> : std::true_type {};
template <typename T, typename SFINAE = void> struct move_always : std::false_type {};
template <typename T> struct move_always<T, typename std::enable_if<
        move_is_plain_type<T>::value &&
        !std::is_copy_constructible<T>::value && std::is_move_constructible<T>::value &&
        std::is_same<decltype(std::declval<type_caster<T>>().operator T&()), T&>::value
    >::type> : std::true_type {};
template <typename T, typename SFINAE = void> struct move_if_unreferenced : std::false_type {};
template <typename T> struct move_if_unreferenced<T, typename std::enable_if<
        move_is_plain_type<T>::value &&
        !move_always<T>::value && std::is_move_constructible<T>::value &&
        std::is_same<decltype(std::declval<type_caster<T>>().operator T&()), T&>::value
    >::type> : std::true_type {};
template <typename T> using move_never = std::integral_constant<bool, !move_always<T>::value && !move_if_unreferenced<T>::value>;

// Detect whether returning a `type` from a cast on type's type_caster is going to result in a
// reference or pointer to a local variable of the type_caster.  Basically, only
// non-reference/pointer `type`s and reference/pointers from a type_caster_generic are safe;
// everything else returns a reference/pointer to a local variable.
template <typename type> using cast_is_temporary_value_reference = bool_constant<
    (std::is_reference<type>::value || std::is_pointer<type>::value) &&
    !std::is_base_of<type_caster_generic, make_caster<type>>::value
>;


NAMESPACE_END(detail)

template <typename T> T cast(const handle &handle) {
    using type_caster = detail::make_caster<T>;
    static_assert(!detail::cast_is_temporary_value_reference<T>::value,
            "Unable to cast type to reference: value is local to type caster");
    type_caster conv;
    if (!conv.load(handle, true)) {
#if defined(NDEBUG)
        throw cast_error("Unable to cast Python instance to C++ type (compile in debug mode for details)");
#else
        throw cast_error("Unable to cast Python instance of type " +
            (std::string) handle.get_type().str() + " to C++ type '" + type_id<T>() + "''");
#endif
    }
    return conv.operator typename type_caster::template cast_op_type<T>();
}

template <typename T> object cast(const T &value,
        return_value_policy policy = return_value_policy::automatic_reference,
        handle parent = handle()) {
    if (policy == return_value_policy::automatic)
        policy = std::is_pointer<T>::value ? return_value_policy::take_ownership : return_value_policy::copy;
    else if (policy == return_value_policy::automatic_reference)
        policy = std::is_pointer<T>::value ? return_value_policy::reference : return_value_policy::copy;
    return object(detail::make_caster<T>::cast(value, policy, parent), false);
}

template <typename T> T handle::cast() const { return pybind11::cast<T>(*this); }
template <> inline void handle::cast() const { return; }

template <typename T>
typename std::enable_if<detail::move_always<T>::value || detail::move_if_unreferenced<T>::value, T>::type move(object &&obj) {
    if (obj.ref_count() > 1)
#if defined(NDEBUG)
        throw cast_error("Unable to cast Python instance to C++ rvalue: instance has multiple references"
            " (compile in debug mode for details)");
#else
        throw cast_error("Unable to move from Python " + (std::string) obj.get_type().str() +
                " instance to C++ " + type_id<T>() + " instance: instance has multiple references");
#endif

    typedef detail::type_caster<T> type_caster;
    type_caster conv;
    if (!conv.load(obj, true))
#if defined(NDEBUG)
        throw cast_error("Unable to cast Python instance to C++ type (compile in debug mode for details)");
#else
        throw cast_error("Unable to cast Python instance of type " +
            (std::string) obj.get_type().str() + " to C++ type '" + type_id<T>() + "''");
#endif

    // Move into a temporary and return that, because the reference may be a local value of `conv`
    T ret = std::move(conv.operator T&());
    return ret;
}

// Calling cast() on an rvalue calls pybind::cast with the object rvalue, which does:
// - If we have to move (because T has no copy constructor), do it.  This will fail if the moved
//   object has multiple references, but trying to copy will fail to compile.
// - If both movable and copyable, check ref count: if 1, move; otherwise copy
// - Otherwise (not movable), copy.
template <typename T> typename std::enable_if<detail::move_always<T>::value, T>::type cast(object &&object) {
    return move<T>(std::move(object));
}
template <typename T> typename std::enable_if<detail::move_if_unreferenced<T>::value, T>::type cast(object &&object) {
    if (object.ref_count() > 1)
        return cast<T>(object);
    else
        return move<T>(std::move(object));
}
template <typename T> typename std::enable_if<detail::move_never<T>::value, T>::type cast(object &&object) {
    return cast<T>(object);
}

template <typename T> T object::cast() const & { return pybind11::cast<T>(*this); }
template <typename T> T object::cast() && { return pybind11::cast<T>(std::move(*this)); }
template <> inline void object::cast() const & { return; }
template <> inline void object::cast() && { return; }



template <return_value_policy policy = return_value_policy::automatic_reference,
          typename... Args> tuple make_tuple(Args&&... args_) {
    const size_t size = sizeof...(Args);
    std::array<object, size> args {
        { object(detail::make_caster<Args>::cast(
            std::forward<Args>(args_), policy, nullptr), false)... }
    };
    for (auto &arg_value : args) {
        if (!arg_value) {
#if defined(NDEBUG)
            throw cast_error("make_tuple(): unable to convert arguments to Python object (compile in debug mode for details)");
#else
            throw cast_error("make_tuple(): unable to convert arguments of types '" +
                (std::string) type_id<std::tuple<Args...>>() + "' to Python object");
#endif
        }
    }
    tuple result(size);
    int counter = 0;
    for (auto &arg_value : args)
        PyTuple_SET_ITEM(result.ptr(), counter++, arg_value.release().ptr());
    return result;
}

/// Annotation for keyword arguments
struct arg {
    constexpr explicit arg(const char *name) : name(name) { }
    template <typename T> arg_v operator=(T &&value) const;

    const char *name;
};

/// Annotation for keyword arguments with values
struct arg_v : arg {
    template <typename T>
    arg_v(const char *name, T &&x, const char *descr = nullptr)
        : arg(name),
          value(detail::make_caster<T>::cast(x, return_value_policy::automatic, handle()), false),
          descr(descr)
#if !defined(NDEBUG)
        , type(type_id<T>())
#endif
    { }

    object value;
    const char *descr;
#if !defined(NDEBUG)
    std::string type;
#endif
};

template <typename T>
arg_v arg::operator=(T &&value) const { return {name, std::forward<T>(value)}; }

/// Alias for backward compatibility -- to be remove in version 2.0
template <typename /*unused*/> using arg_t = arg_v;

inline namespace literals {
/// String literal version of arg
constexpr arg operator"" _a(const char *name, size_t) { return arg(name); }
}

NAMESPACE_BEGIN(detail)
NAMESPACE_BEGIN(constexpr_impl)
/// Implementation details for constexpr functions
constexpr int first(int i) { return i; }
template <typename T, typename... Ts>
constexpr int first(int i, T v, Ts... vs) { return v ? i : first(i + 1, vs...); }

constexpr int last(int /*i*/, int result) { return result; }
template <typename T, typename... Ts>
constexpr int last(int i, int result, T v, Ts... vs) { return last(i + 1, v ? i : result, vs...); }
NAMESPACE_END(constexpr_impl)

/// Return the index of the first type in Ts which satisfies Predicate<T>
template <template<typename> class Predicate, typename... Ts>
constexpr int constexpr_first() { return constexpr_impl::first(0, Predicate<Ts>::value...); }

/// Return the index of the last type in Ts which satisfies Predicate<T>
template <template<typename> class Predicate, typename... Ts>
constexpr int constexpr_last() { return constexpr_impl::last(0, -1, Predicate<Ts>::value...); }

/// Helper class which collects only positional arguments for a Python function call.
/// A fancier version below can collect any argument, but this one is optimal for simple calls.
template <return_value_policy policy>
class simple_collector {
public:
    template <typename... Ts>
    simple_collector(Ts &&...values)
        : m_args(pybind11::make_tuple<policy>(std::forward<Ts>(values)...)) { }

    const tuple &args() const & { return m_args; }
    dict kwargs() const { return {}; }

    tuple args() && { return std::move(m_args); }

    /// Call a Python function and pass the collected arguments
    object call(PyObject *ptr) const {
        auto result = object(PyObject_CallObject(ptr, m_args.ptr()), false);
        if (!result)
            throw error_already_set();
        return result;
    }

private:
    tuple m_args;
};

/// Helper class which collects positional, keyword, * and ** arguments for a Python function call
template <return_value_policy policy>
class unpacking_collector {
public:
    template <typename... Ts>
    unpacking_collector(Ts &&...values) {
        // Tuples aren't (easily) resizable so a list is needed for collection,
        // but the actual function call strictly requires a tuple.
        auto args_list = list();
        int _[] = { 0, (process(args_list, std::forward<Ts>(values)), 0)... };
        ignore_unused(_);

        m_args = object(PyList_AsTuple(args_list.ptr()), false);
    }

    const tuple &args() const & { return m_args; }
    const dict &kwargs() const & { return m_kwargs; }

    tuple args() && { return std::move(m_args); }
    dict kwargs() && { return std::move(m_kwargs); }

    /// Call a Python function and pass the collected arguments
    object call(PyObject *ptr) const {
        auto result = object(PyObject_Call(ptr, m_args.ptr(), m_kwargs.ptr()), false);
        if (!result)
            throw error_already_set();
        return result;
    }

private:
    template <typename T>
    void process(list &args_list, T &&x) {
        auto o = object(detail::make_caster<T>::cast(std::forward<T>(x), policy, nullptr), false);
        if (!o) {
#if defined(NDEBUG)
            argument_cast_error();
#else
            argument_cast_error(std::to_string(args_list.size()), type_id<T>());
#endif
        }
        args_list.append(o);
    }

    void process(list &args_list, detail::args_proxy ap) {
        for (const auto &a : ap) {
            args_list.append(a.cast<object>());
        }
    }

    void process(list &/*args_list*/, arg_v a) {
        if (m_kwargs[a.name]) {
#if defined(NDEBUG)
            multiple_values_error();
#else
            multiple_values_error(a.name);
#endif
        }
        if (!a.value) {
#if defined(NDEBUG)
            argument_cast_error();
#else
            argument_cast_error(a.name, a.type);
#endif
        }
        m_kwargs[a.name] = a.value;
    }

    void process(list &/*args_list*/, detail::kwargs_proxy kp) {
        for (const auto &k : dict(kp, true)) {
            if (m_kwargs[k.first]) {
#if defined(NDEBUG)
                multiple_values_error();
#else
                multiple_values_error(k.first.str());
#endif
            }
            m_kwargs[k.first] = k.second;
        }
    }

    [[noreturn]] static void multiple_values_error() {
        throw type_error("Got multiple values for keyword argument "
                         "(compile in debug mode for details)");
    }

    [[noreturn]] static void multiple_values_error(std::string name) {
        throw type_error("Got multiple values for keyword argument '" + name + "'");
    }

    [[noreturn]] static void argument_cast_error() {
        throw cast_error("Unable to convert call argument to Python object "
                         "(compile in debug mode for details)");
    }

    [[noreturn]] static void argument_cast_error(std::string name, std::string type) {
        throw cast_error("Unable to convert call argument '" + name
                         + "' of type '" + type + "' to Python object");
    }

private:
    tuple m_args;
    dict m_kwargs;
};

/// Collect only positional arguments for a Python function call
template <return_value_policy policy, typename... Args,
          typename = enable_if_t<all_of_t<is_positional, Args...>::value>>
simple_collector<policy> collect_arguments(Args &&...args) {
    return {std::forward<Args>(args)...};
}

/// Collect all arguments, including keywords and unpacking (only instantiated when needed)
template <return_value_policy policy, typename... Args,
          typename = enable_if_t<!all_of_t<is_positional, Args...>::value>>
unpacking_collector<policy> collect_arguments(Args &&...args) {
    // Following argument order rules for generalized unpacking according to PEP 448
    static_assert(
        constexpr_last<is_positional, Args...>() < constexpr_first<is_keyword_or_ds, Args...>()
        && constexpr_last<is_s_unpacking, Args...>() < constexpr_first<is_ds_unpacking, Args...>(),
        "Invalid function call: positional args must precede keywords and ** unpacking; "
        "* unpacking must precede ** unpacking"
    );
    return {std::forward<Args>(args)...};
}

NAMESPACE_END(detail)

template <return_value_policy policy, typename... Args>
object handle::operator()(Args &&...args) const {
    return detail::collect_arguments<policy>(std::forward<Args>(args)...).call(m_ptr);
}

template <return_value_policy policy,
          typename... Args> object handle::call(Args &&... args) const {
    return operator()<policy>(std::forward<Args>(args)...);
}

#define PYBIND11_MAKE_OPAQUE(Type) \
    namespace pybind11 { namespace detail { \
        template<> class type_caster<Type> : public type_caster_base<Type> { }; \
    }}

NAMESPACE_END(pybind11)
