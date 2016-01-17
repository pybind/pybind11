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

            NPY_C_CONTIGUOUS_ = 0x0001,
            NPY_F_CONTIGUOUS_ = 0x0002,
            NPY_ARRAY_FORCECAST_ = 0x0010,
            NPY_ENSURE_ARRAY_ = 0x0040,
            NPY_BOOL_ = 0,
            NPY_BYTE_, NPY_UBYTE_,
            NPY_SHORT_, NPY_USHORT_,
            NPY_INT_, NPY_UINT_,
            NPY_LONG_, NPY_ULONG_,
            NPY_LONGLONG_, NPY_ULONGLONG_,
            NPY_FLOAT_, NPY_DOUBLE_, NPY_LONGDOUBLE_,
            NPY_CFLOAT_, NPY_CDOUBLE_, NPY_CLONGDOUBLE_
        };

        static API lookup() {
            module m = module::import("numpy.core.multiarray");
            object c = (object) m.attr("_ARRAY_API");
#if PY_MAJOR_VERSION >= 3
            void **api_ptr = (void **) (c ? PyCapsule_GetPointer(c.ptr(), NULL) : nullptr);
#else
            void **api_ptr = (void **) (c ? PyCObject_AsVoidPtr(c.ptr()) : nullptr);
#endif
            API api;
            api.PyArray_Type_          = (decltype(api.PyArray_Type_))          api_ptr[API_PyArray_Type];
            api.PyArray_DescrFromType_ = (decltype(api.PyArray_DescrFromType_)) api_ptr[API_PyArray_DescrFromType];
            api.PyArray_FromAny_       = (decltype(api.PyArray_FromAny_))       api_ptr[API_PyArray_FromAny];
            api.PyArray_NewCopy_       = (decltype(api.PyArray_NewCopy_))       api_ptr[API_PyArray_NewCopy];
            api.PyArray_NewFromDescr_  = (decltype(api.PyArray_NewFromDescr_))  api_ptr[API_PyArray_NewFromDescr];
            return api;
        }

        bool PyArray_Check_(PyObject *obj) const { return (bool) PyObject_TypeCheck(obj, PyArray_Type_); }

        PyObject *(*PyArray_DescrFromType_)(int);
        PyObject *(*PyArray_NewFromDescr_)
            (PyTypeObject *, PyObject *, int, Py_intptr_t *,
             Py_intptr_t *, void *, int, PyObject *);
        PyObject *(*PyArray_NewCopy_)(PyObject *, int);
        PyTypeObject *PyArray_Type_;
        PyObject *(*PyArray_FromAny_) (PyObject *, PyObject *, int, int, int, PyObject *);
    };

    PYBIND11_OBJECT_DEFAULT(array, buffer, lookup_api().PyArray_Check_)

    template <typename Type> array(size_t size, const Type *ptr) {
        API& api = lookup_api();
        PyObject *descr = api.PyArray_DescrFromType_(npy_format_descriptor<Type>::value);
        if (descr == nullptr)
            pybind11_fail("NumPy: unsupported buffer format!");
        Py_intptr_t shape = (Py_intptr_t) size;
        object tmp = object(api.PyArray_NewFromDescr_(
            api.PyArray_Type_, descr, 1, &shape, nullptr, (void *) ptr, 0, nullptr), false);
        if (ptr && tmp)
            tmp = object(api.PyArray_NewCopy_(tmp.ptr(), -1 /* any order */), false);
        if (!tmp)
            pybind11_fail("NumPy: unable to create array!");
        m_ptr = tmp.release().ptr();
    }

    array(const buffer_info &info) {
        API& api = lookup_api();
        if ((info.format.size() < 1) || (info.format.size() > 2))
            pybind11_fail("Unsupported buffer format!");
        int fmt = (int) info.format[0];
        if (info.format == "Zd")      fmt = API::NPY_CDOUBLE_;
        else if (info.format == "Zf") fmt = API::NPY_CFLOAT_;

        PyObject *descr = api.PyArray_DescrFromType_(fmt);
        if (descr == nullptr)
            pybind11_fail("NumPy: unsupported buffer format '" + info.format + "'!");
        object tmp(api.PyArray_NewFromDescr_(
            api.PyArray_Type_, descr, info.ndim, (Py_intptr_t *) &info.shape[0],
            (Py_intptr_t *) &info.strides[0], info.ptr, 0, nullptr), false);
        if (info.ptr && tmp)
            tmp = object(api.PyArray_NewCopy_(tmp.ptr(), -1 /* any order */), false);
        if (!tmp)
            pybind11_fail("NumPy: unable to create array!");
        m_ptr = tmp.release().ptr();
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
    static PyObject *ensure(PyObject *ptr) {
        if (ptr == nullptr)
            return nullptr;
        API &api = lookup_api();
        PyObject *descr = api.PyArray_DescrFromType_(npy_format_descriptor<T>::value);
        PyObject *result = api.PyArray_FromAny_(
            ptr, descr, 0, 0, API::NPY_C_CONTIGUOUS_ | API::NPY_ENSURE_ARRAY_
                            | API::NPY_ARRAY_FORCECAST_, nullptr);
        Py_DECREF(ptr);
        return result;
    }
};

#define DECL_FMT(t, n) template<> struct npy_format_descriptor<t> { enum { value = array::API::n }; }
DECL_FMT(int8_t, NPY_BYTE_);  DECL_FMT(uint8_t, NPY_UBYTE_); DECL_FMT(int16_t, NPY_SHORT_);
DECL_FMT(uint16_t, NPY_USHORT_); DECL_FMT(int32_t, NPY_INT_); DECL_FMT(uint32_t, NPY_UINT_);
DECL_FMT(int64_t, NPY_LONGLONG_); DECL_FMT(uint64_t, NPY_ULONGLONG_); DECL_FMT(float, NPY_FLOAT_);
DECL_FMT(double, NPY_DOUBLE_); DECL_FMT(bool, NPY_BOOL_); DECL_FMT(std::complex<float>, NPY_CFLOAT_);
DECL_FMT(std::complex<double>, NPY_CDOUBLE_);
#undef DECL_FMT

NAMESPACE_BEGIN(detail)

template <typename T> struct handle_type_name<array_t<T>> {
    static PYBIND11_DESCR name() { return _("array[") + type_caster<T>::name() + _("]"); }
};

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
        int ndim = 0; size_t size = 0;
        std::vector<size_t> shape;
        for (size_t i=0; i<N; ++i) {
            if (buffers[i].size > size) {
                ndim = buffers[i].ndim;
                shape = buffers[i].shape;
                size = buffers[i].size;
            }
        }
        std::vector<size_t> strides(ndim);
        if (ndim > 0) {
            strides[ndim-1] = sizeof(Return);
            for (int i=ndim-1; i>0; --i)
                strides[i-1] = strides[i] * shape[i];
        }

        /* Check if the parameters are actually compatible */
        for (size_t i=0; i<N; ++i)
            if (buffers[i].size != 1 && (buffers[i].ndim != ndim || buffers[i].shape != shape))
                pybind11_fail("pybind11::vectorize: incompatible size/dimension of inputs!");

        if (size == 1)
            return cast(f(*((Args *) buffers[Index].ptr)...));

        array result(buffer_info(nullptr, sizeof(Return),
            format_descriptor<Return>::value(),
            ndim, shape, strides));

        buffer_info buf = result.request();
        Return *output = (Return *) buf.ptr;

        /* Call the function */
        for (size_t i=0; i<size; ++i)
            output[i] = f((buffers[Index].size == 1
                               ? *((Args *) buffers[Index].ptr)
                               : ((Args *) buffers[Index].ptr)[i])...);

        return result;
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
