/*
    pybind11/numpy.h: Basic NumPy support, vectorize() wrapper

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "pybind11.h"
#include "complex.h"
#include <numeric>
#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#endif

NAMESPACE_BEGIN(pybind11)
namespace detail { template <typename type, typename SFINAE = void> struct npy_format_descriptor { }; }

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

    enum {
        c_style = API::NPY_C_CONTIGUOUS_,
        f_style = API::NPY_F_CONTIGUOUS_,
        forcecast = API::NPY_ARRAY_FORCECAST_
    };

    template <typename Type> array(size_t size, const Type *ptr) {
        API& api = lookup_api();
        PyObject *descr = api.PyArray_DescrFromType_(detail::npy_format_descriptor<Type>::value);
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
            api.PyArray_Type_, descr, (int) info.ndim, (Py_intptr_t *) &info.shape[0],
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

template <typename T, int ExtraFlags = array::forcecast> class array_t : public array {
public:
    PYBIND11_OBJECT_CVT(array_t, array, is_non_null, m_ptr = ensure(m_ptr));
    array_t() : array() { }
    array_t(const buffer_info& info) : array(info) {}
    static bool is_non_null(PyObject *ptr) { return ptr != nullptr; }
    static PyObject *ensure(PyObject *ptr) {
        if (ptr == nullptr)
            return nullptr;
        API &api = lookup_api();
        PyObject *descr = api.PyArray_DescrFromType_(detail::npy_format_descriptor<T>::value);
        PyObject *result = api.PyArray_FromAny_(ptr, descr, 0, 0, API::NPY_ENSURE_ARRAY_ | ExtraFlags, nullptr);
        if (!result)
            PyErr_Clear();
        Py_DECREF(ptr);
        return result;
    }
};

NAMESPACE_BEGIN(detail)

template <typename T> struct npy_format_descriptor<T, typename std::enable_if<std::is_integral<T>::value>::type> {
private:
    constexpr static const int values[8] = {
        array::API::NPY_BYTE_, array::API::NPY_UBYTE_, array::API::NPY_SHORT_,    array::API::NPY_USHORT_,
        array::API::NPY_INT_,  array::API::NPY_UINT_,  array::API::NPY_LONGLONG_, array::API::NPY_ULONGLONG_ };
public:
    enum { value = values[detail::log2(sizeof(T)) * 2 + (std::is_unsigned<T>::value ? 1 : 0)] };
    template <typename T2 = T, typename std::enable_if<std::is_signed<T2>::value, int>::type = 0>
    static PYBIND11_DESCR name() { return _("int") + _<sizeof(T)*8>(); }
    template <typename T2 = T, typename std::enable_if<!std::is_signed<T2>::value, int>::type = 0>
    static PYBIND11_DESCR name() { return _("uint") + _<sizeof(T)*8>(); }
};
template <typename T> constexpr const int npy_format_descriptor<
    T, typename std::enable_if<std::is_integral<T>::value>::type>::values[8];

#define DECL_FMT(Type, NumPyName, Name) template<> struct npy_format_descriptor<Type> { \
    enum { value = array::API::NumPyName }; \
    static PYBIND11_DESCR name() { return _(Name); } }
DECL_FMT(float, NPY_FLOAT_, "float32"); DECL_FMT(double, NPY_DOUBLE_, "float64"); DECL_FMT(bool, NPY_BOOL_, "bool");
DECL_FMT(std::complex<float>, NPY_CFLOAT_, "complex64"); DECL_FMT(std::complex<double>, NPY_CDOUBLE_, "complex128");
#undef DECL_FMT

template  <class T>
using array_iterator = typename std::add_pointer<T>::type;

template <class T>
array_iterator<T> array_begin(const buffer_info& buffer) {
    return array_iterator<T>(reinterpret_cast<T*>(buffer.ptr));
}

template <class T>
array_iterator<T> array_end(const buffer_info& buffer) {
    return array_iterator<T>(reinterpret_cast<T*>(buffer.ptr) + buffer.size);
}

class common_iterator {
public:
    using container_type = std::vector<size_t>;
    using value_type = container_type::value_type;
    using size_type = container_type::size_type;

    common_iterator() : p_ptr(0), m_strides() {}

    common_iterator(void* ptr, const container_type& strides, const std::vector<size_t>& shape)
        : p_ptr(reinterpret_cast<char*>(ptr)), m_strides(strides.size()) {
        m_strides.back() = static_cast<value_type>(strides.back());
        for (size_type i = m_strides.size() - 1; i != 0; --i) {
            size_type j = i - 1;
            value_type s = static_cast<value_type>(shape[i]);
            m_strides[j] = strides[j] + m_strides[i] - strides[i] * s;
        }
    }

    void increment(size_type dim) {
        p_ptr += m_strides[dim];
    }

    void* data() const {
        return p_ptr;
    }

private:
    char* p_ptr;
    container_type m_strides;
};

template <size_t N> class multi_array_iterator {
public:
    using container_type = std::vector<size_t>;

    multi_array_iterator(const std::array<buffer_info, N> &buffers,
                         const std::vector<size_t> &shape)
        : m_shape(shape.size()), m_index(shape.size(), 0),
          m_common_iterator() {

        // Manual copy to avoid conversion warning if using std::copy
        for (size_t i = 0; i < shape.size(); ++i)
            m_shape[i] = static_cast<container_type::value_type>(shape[i]);

        container_type strides(shape.size());
        for (size_t i = 0; i < N; ++i)
            init_common_iterator(buffers[i], shape, m_common_iterator[i], strides);
    }

    multi_array_iterator& operator++() {
        for (size_t j = m_index.size(); j != 0; --j) {
            size_t i = j - 1;
            if (++m_index[i] != m_shape[i]) {
                increment_common_iterator(i);
                break;
            } else {
                m_index[i] = 0;
            }
        }
        return *this;
    }

    template <size_t K, class T> const T& data() const {
        return *reinterpret_cast<T*>(m_common_iterator[K].data());
    }

private:

    using common_iter = common_iterator;

    void init_common_iterator(const buffer_info &buffer,
                              const std::vector<size_t> &shape,
                              common_iter &iterator, container_type &strides) {
        auto buffer_shape_iter = buffer.shape.rbegin();
        auto buffer_strides_iter = buffer.strides.rbegin();
        auto shape_iter = shape.rbegin();
        auto strides_iter = strides.rbegin();

        while (buffer_shape_iter != buffer.shape.rend()) {
            if (*shape_iter == *buffer_shape_iter)
                *strides_iter = static_cast<size_t>(*buffer_strides_iter);
            else
                *strides_iter = 0;

            ++buffer_shape_iter;
            ++buffer_strides_iter;
            ++shape_iter;
            ++strides_iter;
        }

        std::fill(strides_iter, strides.rend(), 0);
        iterator = common_iter(buffer.ptr, strides, shape);
    }

    void increment_common_iterator(size_t dim) {
        for (auto &iter : m_common_iterator)
            iter.increment(dim);
    }

    container_type m_shape;
    container_type m_index;
    std::array<common_iter, N> m_common_iterator;
};

template <size_t N>
bool broadcast(const std::array<buffer_info, N>& buffers, size_t& ndim, std::vector<size_t>& shape) {
    ndim = std::accumulate(buffers.begin(), buffers.end(), size_t(0), [](size_t res, const buffer_info& buf) {
        return std::max(res, buf.ndim);
    });

    shape = std::vector<size_t>(ndim, 1);
    bool trivial_broadcast = true;
    for (size_t i = 0; i < N; ++i) {
        auto res_iter = shape.rbegin();
        bool i_trivial_broadcast = (buffers[i].size == 1) || (buffers[i].ndim == ndim);
        for (auto shape_iter = buffers[i].shape.rbegin();
             shape_iter != buffers[i].shape.rend(); ++shape_iter, ++res_iter) {

            if (*res_iter == 1)
                *res_iter = *shape_iter;
            else if ((*shape_iter != 1) && (*res_iter != *shape_iter))
                pybind11_fail("pybind11::vectorize: incompatible size/dimension of inputs!");

            i_trivial_broadcast = i_trivial_broadcast && (*res_iter == *shape_iter);
        }
        trivial_broadcast = trivial_broadcast && i_trivial_broadcast;
    }
    return trivial_broadcast;
}

template <typename Func, typename Return, typename... Args>
struct vectorize_helper {
    typename std::remove_reference<Func>::type f;

    template <typename T>
    vectorize_helper(T&&f) : f(std::forward<T>(f)) { }

    object operator()(array_t<Args, array::c_style | array::forcecast>... args) {
        return run(args..., typename make_index_sequence<sizeof...(Args)>::type());
    }

    template <size_t ... Index> object run(array_t<Args, array::c_style | array::forcecast>&... args, index_sequence<Index...> index) {
        /* Request buffers from all parameters */
        const size_t N = sizeof...(Args);

        std::array<buffer_info, N> buffers {{ args.request()... }};

        /* Determine dimensions parameters of output array */
        size_t ndim = 0;
        std::vector<size_t> shape(0);
        bool trivial_broadcast = broadcast(buffers, ndim, shape);

        size_t size = 1;
        std::vector<size_t> strides(ndim);
        if (ndim > 0) {
            strides[ndim-1] = sizeof(Return);
            for (size_t i = ndim - 1; i > 0; --i) {
                strides[i - 1] = strides[i] * shape[i];
                size *= shape[i];
            }
            size *= shape[0];
        }

        if (size == 1)
            return cast(f(*((Args *) buffers[Index].ptr)...));

        array result(buffer_info(nullptr, sizeof(Return),
            format_descriptor<Return>::value,
            ndim, shape, strides));

        buffer_info buf = result.request();
        Return *output = (Return *) buf.ptr;

        if (trivial_broadcast) {
            /* Call the function */
            for (size_t i=0; i<size; ++i) {
                output[i] = f((buffers[Index].size == 1
                               ? *((Args *) buffers[Index].ptr)
                               : ((Args *) buffers[Index].ptr)[i])...);
            }
        } else {
            apply_broadcast<N, Index...>(buffers, buf, index);
        }

        return result;
    }

    template <size_t N, size_t... Index>
    void apply_broadcast(const std::array<buffer_info, N> &buffers,
                         buffer_info &output, index_sequence<Index...>) {
        using input_iterator = multi_array_iterator<N>;
        using output_iterator = array_iterator<Return>;

        input_iterator input_iter(buffers, output.shape);
        output_iterator output_end = array_end<Return>(output);

        for (output_iterator iter = array_begin<Return>(output);
             iter != output_end; ++iter, ++input_iter) {
            *iter = f((input_iter.template data<Index, Args>())...);
        }
    }
};

template <typename T, int Flags> struct handle_type_name<array_t<T, Flags>> {
    static PYBIND11_DESCR name() { return _("numpy.ndarray[dtype=") + type_caster<T>::name() + _("]"); }
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
