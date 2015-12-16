/*
    pybind11/numpy.h: Basic NumPy support, auto-vectorization support

    Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "pybind11.h"
#include "complex.h"

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#endif

NAMESPACE_BEGIN(pybind11)

template <typename type> struct npy_format_descriptor { };

class array : public buffer {
public:
    struct API {
        enum Entries {
            API_PyArray_Type = 2,
            API_PyArray_DescrFromType = 45,
            API_PyArray_FromAny = 69,
            API_PyArray_NewCopy = 85,
            API_PyArray_NewFromDescr = 94,
            NPY_C_CONTIGUOUS = 0x0001,
            NPY_F_CONTIGUOUS = 0x0002,
            NPY_NPY_ARRAY_FORCECAST = 0x0010,
            NPY_ENSURE_ARRAY = 0x0040,
            NPY_BOOL=0,
            NPY_BYTE, NPY_UBYTE,
            NPY_SHORT, NPY_USHORT,
            NPY_INT, NPY_UINT,
            NPY_LONG, NPY_ULONG,
            NPY_LONGLONG, NPY_ULONGLONG,
            NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
            NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE
        };

        static API lookup() {
            PyObject *numpy = PyImport_ImportModule("numpy.core.multiarray");
            PyObject *capsule = numpy ? PyObject_GetAttrString(numpy, "_ARRAY_API") : nullptr;
#if PY_MAJOR_VERSION >= 3
            void **api_ptr = (void **) (capsule ? PyCapsule_GetPointer(capsule, NULL) : nullptr);
#else
            void **api_ptr = (void **) (capsule ? PyCObject_AsVoidPtr(capsule) : nullptr);
#endif
            Py_XDECREF(capsule);
            Py_XDECREF(numpy);
            if (api_ptr == nullptr)
                throw std::runtime_error("Could not acquire pointer to NumPy API!");
            API api;
            api.PyArray_Type          = (decltype(api.PyArray_Type))          api_ptr[API_PyArray_Type];
            api.PyArray_DescrFromType = (decltype(api.PyArray_DescrFromType)) api_ptr[API_PyArray_DescrFromType];
            api.PyArray_FromAny       = (decltype(api.PyArray_FromAny))       api_ptr[API_PyArray_FromAny];
            api.PyArray_NewCopy       = (decltype(api.PyArray_NewCopy))       api_ptr[API_PyArray_NewCopy];
            api.PyArray_NewFromDescr  = (decltype(api.PyArray_NewFromDescr))  api_ptr[API_PyArray_NewFromDescr];
            return api;
        }

        bool PyArray_Check(PyObject *obj) const { return (bool) PyObject_TypeCheck(obj, PyArray_Type); }

        PyObject *(*PyArray_DescrFromType)(int);
        PyObject *(*PyArray_NewFromDescr)
            (PyTypeObject *, PyObject *, int, Py_intptr_t *,
             Py_intptr_t *, void *, int, PyObject *);
        PyObject *(*PyArray_NewCopy)(PyObject *, int);
        PyTypeObject *PyArray_Type;
        PyObject *(*PyArray_FromAny) (PyObject *, PyObject *, int, int, int, PyObject *);
    };

    PYBIND11_OBJECT_DEFAULT(array, buffer, lookup_api().PyArray_Check)

    template <typename Type> array(size_t size, const Type *ptr) {
        API& api = lookup_api();
        PyObject *descr = api.PyArray_DescrFromType(npy_format_descriptor<Type>::value);
        if (descr == nullptr)
            throw std::runtime_error("NumPy: unsupported buffer format!");
        Py_intptr_t shape = (Py_intptr_t) size;
        PyObject *tmp = api.PyArray_NewFromDescr(
            api.PyArray_Type, descr, 1, &shape, nullptr, (void *) ptr, 0, nullptr);
        if (tmp == nullptr)
            throw std::runtime_error("NumPy: unable to create array!");
        m_ptr = api.PyArray_NewCopy(tmp, -1 /* any order */);
        Py_DECREF(tmp);
        if (m_ptr == nullptr)
            throw std::runtime_error("NumPy: unable to copy array!");
    }

    array(const buffer_info &info) {
        API& api = lookup_api();
        if ((info.format.size() < 1) || (info.format.size() > 2))
            throw std::runtime_error("Unsupported buffer format!");
        int fmt = (int) info.format[0];
        if (info.format == "Zd")
            fmt = API::NPY_CDOUBLE;
        else if (info.format == "Zf")
            fmt = API::NPY_CFLOAT;
        PyObject *descr = api.PyArray_DescrFromType(fmt);
        if (descr == nullptr)
            throw std::runtime_error("NumPy: unsupported buffer format '" + info.format + "'!");
        PyObject *tmp = api.PyArray_NewFromDescr(
            api.PyArray_Type, descr, info.ndim, (Py_intptr_t *) &info.shape[0],
            (Py_intptr_t *) &info.strides[0], info.ptr, 0, nullptr);
        if (tmp == nullptr)
            throw std::runtime_error("NumPy: unable to create array!");
        m_ptr = api.PyArray_NewCopy(tmp, -1 /* any order */);
        Py_DECREF(tmp);
        if (m_ptr == nullptr)
            throw std::runtime_error("NumPy: unable to copy array!");
    }

protected:
    static API &lookup_api() {
        static API api = API::lookup();
        return api;
    }
};

template <typename T> class array_t : public array {
public:
    PYBIND11_OBJECT_CVT(array_t, array, is_non_null, m_ptr = ensure(m_ptr));
    array_t() : array() { }
    static bool is_non_null(PyObject *ptr) { return ptr != nullptr; }
    PyObject *ensure(PyObject *ptr) {
        if (ptr == nullptr)
            return nullptr;
        API &api = lookup_api();
        PyObject *descr = api.PyArray_DescrFromType(npy_format_descriptor<T>::value);
        return api.PyArray_FromAny(ptr, descr, 0, 0,
                                   API::NPY_C_CONTIGUOUS | API::NPY_ENSURE_ARRAY |
                                   API::NPY_NPY_ARRAY_FORCECAST, nullptr);
    }
};

#define DECL_FMT(t, n) template<> struct npy_format_descriptor<t> { enum { value = array::API::n }; }
DECL_FMT(int8_t, NPY_BYTE);  DECL_FMT(uint8_t, NPY_UBYTE); DECL_FMT(int16_t, NPY_SHORT);
DECL_FMT(uint16_t, NPY_USHORT); DECL_FMT(int32_t, NPY_INT); DECL_FMT(uint32_t, NPY_UINT);
DECL_FMT(int64_t, NPY_LONGLONG); DECL_FMT(uint64_t, NPY_ULONGLONG); DECL_FMT(float, NPY_FLOAT);
DECL_FMT(double, NPY_DOUBLE); DECL_FMT(bool, NPY_BOOL); DECL_FMT(std::complex<float>, NPY_CFLOAT);
DECL_FMT(std::complex<double>, NPY_CDOUBLE);
#undef DECL_FMT

NAMESPACE_BEGIN(detail)

template <typename Func, typename Return, typename... Args>
struct vectorize_helper {
    typename std::remove_reference<Func>::type f;

    template <typename T>
    vectorize_helper(T&&f) : f(std::forward<T>(f)) { }

    object operator()(array_t<Args>... args) {
        return run(args..., typename make_index_sequence<sizeof...(Args)>::type());
    }

    template <size_t ... Index> object run(array_t<Args>&... args, index_sequence<Index...>) {
        /* Request buffers from all parameters */
        const size_t N = sizeof...(Args);
        std::array<buffer_info, N> buffers {{ args.request()... }};

        /* Determine dimensions parameters of output array */
        int ndim = 0; size_t count = 0;
        std::vector<size_t> shape;
        for (size_t i=0; i<N; ++i) {
            if (buffers[i].count > count) {
                ndim = buffers[i].ndim;
                shape = buffers[i].shape;
                count = buffers[i].count;
            }
        }
        std::vector<size_t> strides(ndim);
        if (ndim > 0) {
            strides[ndim-1] = sizeof(Return);
            for (int i=ndim-1; i>0; --i)
                strides[i-1] = strides[i] * shape[i];
        }

        /* Check if the parameters are actually compatible */
        for (size_t i=0; i<N; ++i) {
            if (buffers[i].count != 1 && (buffers[i].ndim != ndim || buffers[i].shape != shape))
                throw std::runtime_error("pybind11::vectorize: incompatible size/dimension of inputs!");
        }

        /* Call the function */
        std::vector<Return> result(count);
        for (size_t i=0; i<count; ++i)
            result[i] = f((buffers[Index].count == 1
                               ? *((Args *) buffers[Index].ptr)
                               :  ((Args *) buffers[Index].ptr)[i])...);

        if (count == 1)
            return cast(result[0]);

        /* Return the result */
        return array(buffer_info(result.data(), sizeof(Return),
            format_descriptor<Return>::value(),
            ndim, shape, strides));
    }
};

NAMESPACE_END(detail)

template <typename Func, typename Return, typename... Args>
detail::vectorize_helper<Func, Return, Args...> vectorize(const Func &f, Return (*) (Args ...)) {
    return detail::vectorize_helper<Func, Return, Args...>(f);
}

template <typename Return, typename... Args>
detail::vectorize_helper<Return (*) (Args ...), Return, Args...> vectorize(Return (*f) (Args ...)) {
    return vectorize<Return (*) (Args ...), Return, Args...>(f, f);
}

template <typename func> auto vectorize(func &&f) -> decltype(
        vectorize(std::forward<func>(f), (typename detail::remove_class<decltype(&std::remove_reference<func>::type::operator())>::type *) nullptr)) {
    return vectorize(std::forward<func>(f), (typename detail::remove_class<decltype(
                   &std::remove_reference<func>::type::operator())>::type *) nullptr);
}

NAMESPACE_END(pybind11)

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
