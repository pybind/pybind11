/*
    pybind/numpy.h: Basic NumPy support, auto-vectorization support

    Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include <pybind/pybind.h>
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#endif

NAMESPACE_BEGIN(pybind)

class array : public buffer {
protected:
    struct API {
        enum Entries {
            API_PyArray_Type = 2,
            API_PyArray_DescrFromType = 45,
            API_PyArray_FromAny = 69,
            API_PyArray_NewCopy = 85,
            API_PyArray_NewFromDescr = 94,
            API_NPY_C_CONTIGUOUS = 0x0001,
            API_NPY_F_CONTIGUOUS = 0x0002,
            API_NPY_NPY_ARRAY_FORCECAST = 0x0010,
            API_NPY_ENSURE_ARRAY = 0x0040
        };

        static API lookup() {
            PyObject *numpy = PyImport_ImportModule("numpy.core.multiarray");
            PyObject *capsule = numpy ? PyObject_GetAttrString(numpy, "_ARRAY_API") : nullptr;
            void **api_ptr = (void **) (capsule ? PyCapsule_GetPointer(capsule, NULL) : nullptr);
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
public:
    PYBIND_OBJECT_DEFAULT(array, buffer, lookup_api().PyArray_Check)

    template <typename Type> array(size_t size, const Type *ptr) {
        API& api = lookup_api();
        PyObject *descr = api.PyArray_DescrFromType(
            (int) format_descriptor<Type>::value()[0]);
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
        if (info.format.size() != 1)
            throw std::runtime_error("Unsupported buffer format!");
        PyObject *descr = api.PyArray_DescrFromType(info.format[0]);
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

template <typename T> class array_dtype : public array {
public:
    PYBIND_OBJECT_CVT(array_dtype, array, is_non_null, m_ptr = ensure(m_ptr));
    array_dtype() : array() { }
    static bool is_non_null(PyObject *ptr) { return ptr != nullptr; }
    static PyObject *ensure(PyObject *ptr) {
        API &api = lookup_api();
        PyObject *descr = api.PyArray_DescrFromType(format_descriptor<T>::value()[0]);
        return api.PyArray_FromAny(ptr, descr, 0, 0,
                                   API::API_NPY_C_CONTIGUOUS | API::API_NPY_ENSURE_ARRAY |
                                   API::API_NPY_NPY_ARRAY_FORCECAST, nullptr);
    }
};

NAMESPACE_BEGIN(detail)
PYBIND_TYPE_CASTER_PYTYPE(array)
PYBIND_TYPE_CASTER_PYTYPE(array_dtype<int8_t>)  PYBIND_TYPE_CASTER_PYTYPE(array_dtype<uint8_t>)
PYBIND_TYPE_CASTER_PYTYPE(array_dtype<int16_t>) PYBIND_TYPE_CASTER_PYTYPE(array_dtype<uint16_t>)
PYBIND_TYPE_CASTER_PYTYPE(array_dtype<int32_t>) PYBIND_TYPE_CASTER_PYTYPE(array_dtype<uint32_t>)
PYBIND_TYPE_CASTER_PYTYPE(array_dtype<int64_t>) PYBIND_TYPE_CASTER_PYTYPE(array_dtype<uint64_t>)
PYBIND_TYPE_CASTER_PYTYPE(array_dtype<float>)   PYBIND_TYPE_CASTER_PYTYPE(array_dtype<double>)
NAMESPACE_END(detail)

template <typename func_type, typename return_type, typename... args_type, size_t... Index>
    std::function<object(array_dtype<args_type>...)>
        vectorize(func_type &&f, return_type (*) (args_type ...),
                  detail::index_sequence<Index...>) {

    return [f](array_dtype<args_type>... args) -> array {
        /* Request buffers from all parameters */
        const size_t N = sizeof...(args_type);
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
            strides[ndim-1] = sizeof(return_type);
            for (int i=ndim-1; i>0; --i)
                strides[i-1] = strides[i] * shape[i];
        }

        /* Check if the parameters are actually compatible */
        for (size_t i=0; i<N; ++i) {
            if (buffers[i].count != 1 && (buffers[i].ndim != ndim || buffers[i].shape != shape))
                throw std::runtime_error("pybind::vectorize: incompatible size/dimension of inputs!");
        }

        /* Call the function */
        std::vector<return_type> result(count);
        for (size_t i=0; i<count; ++i)
            result[i] = f((buffers[Index].count == 1
                               ? *((args_type *) buffers[Index].ptr)
                               :  ((args_type *) buffers[Index].ptr)[i])...);

        if (count == 1)
            return cast(result[0]);

        /* Return the result */
        return array(buffer_info(result.data(), sizeof(return_type), 
            format_descriptor<return_type>::value(),
            ndim, shape, strides));
    };
}

template <typename func_type, typename return_type, typename... args_type>
    std::function<object(array_dtype<args_type>...)>
        vectorize(func_type &&f, return_type (*f_) (args_type ...) = nullptr) {
    return vectorize(f, f_, typename detail::make_index_sequence<sizeof...(args_type)>::type());
}

template <typename return_type, typename... args_type>
std::function<object(array_dtype<args_type>...)> vectorize(return_type (*f) (args_type ...)) {
    return vectorize(f, f);
}

template <typename func> auto vectorize(func &&f) -> decltype(
        vectorize(std::forward<func>(f), (typename detail::remove_class<decltype(&std::remove_reference<func>::type::operator())>::type *) nullptr)) {
    return vectorize(std::forward<func>(f), (typename detail::remove_class<decltype(
                   &std::remove_reference<func>::type::operator())>::type *) nullptr);
}

NAMESPACE_END(pybind)

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
