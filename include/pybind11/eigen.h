/*
    pybind11/eigen.h: Transparent conversion for dense and sparse Eigen matrices

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "numpy.h"

#if defined(__GNUG__) || defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wconversion"
#  pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <Eigen/Core>
#include <Eigen/SparseCore>

#if defined(__GNUG__) || defined(__clang__)
#  pragma GCC diagnostic pop
#endif

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#endif

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

template <typename T> class is_eigen_dense {
private:
    template<typename Derived> static std::true_type test(const Eigen::DenseBase<Derived> &);
    static std::false_type test(...);
public:
    static constexpr bool value = decltype(test(std::declval<T>()))::value;
};

template <typename T> class is_eigen_sparse {
private:
    template<typename Derived> static std::true_type test(const Eigen::SparseMatrixBase<Derived> &);
    static std::false_type test(...);
public:
    static constexpr bool value = decltype(test(std::declval<T>()))::value;
};

template<typename Type>
struct type_caster<Type, typename std::enable_if<is_eigen_dense<Type>::value>::type> {
    typedef typename Type::Scalar Scalar;
    static constexpr bool rowMajor = Type::Flags & Eigen::RowMajorBit;
    static constexpr bool isVector = Type::IsVectorAtCompileTime;

    bool load(handle src, bool) {
       array_t<Scalar> buffer(src, true);
       if (!buffer.check())
           return false;

        buffer_info info = buffer.request();
        if (info.ndim == 1) {
            typedef Eigen::InnerStride<> Strides;
            if (!isVector &&
                !(Type::RowsAtCompileTime == Eigen::Dynamic &&
                  Type::ColsAtCompileTime == Eigen::Dynamic))
                return false;

            if (Type::SizeAtCompileTime != Eigen::Dynamic &&
                info.shape[0] != (size_t) Type::SizeAtCompileTime)
                return false;

            auto strides = Strides(info.strides[0] / sizeof(Scalar));

            Strides::Index n_elts = (Strides::Index) info.shape[0];
            Strides::Index unity = 1;

            value = Eigen::Map<Type, 0, Strides>(
                (Scalar *) info.ptr, rowMajor ? unity : n_elts, rowMajor ? n_elts : unity, strides);
        } else if (info.ndim == 2) {
            typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> Strides;

            if ((Type::RowsAtCompileTime != Eigen::Dynamic && info.shape[0] != (size_t) Type::RowsAtCompileTime) ||
                (Type::ColsAtCompileTime != Eigen::Dynamic && info.shape[1] != (size_t) Type::ColsAtCompileTime))
                return false;

            auto strides = Strides(
                info.strides[rowMajor ? 0 : 1] / sizeof(Scalar),
                info.strides[rowMajor ? 1 : 0] / sizeof(Scalar));

            value = Eigen::Map<Type, 0, Strides>(
                (Scalar *) info.ptr,
                typename Strides::Index(info.shape[0]),
                typename Strides::Index(info.shape[1]), strides);
        } else {
            return false;
        }
        return true;
    }

    static handle cast(const Type &src, return_value_policy /* policy */, handle /* parent */) {
        if (isVector) {
            return array(buffer_info(
                /* Pointer to buffer */
                const_cast<Scalar *>(src.data()),
                /* Size of one scalar */
                sizeof(Scalar),
                /* Python struct-style format descriptor */
                format_descriptor<Scalar>::value,
                /* Number of dimensions */
                1,
                /* Buffer dimensions */
                { (size_t) src.size() },
                /* Strides (in bytes) for each index */
                { sizeof(Scalar) }
            )).release();
        } else {
            return array(buffer_info(
                /* Pointer to buffer */
                const_cast<Scalar *>(src.data()),
                /* Size of one scalar */
                sizeof(Scalar),
                /* Python struct-style format descriptor */
                format_descriptor<Scalar>::value,
                /* Number of dimensions */
                isVector ? 1 : 2,
                /* Buffer dimensions */
                { (size_t) src.rows(),
                  (size_t) src.cols() },
                /* Strides (in bytes) for each index */
                { sizeof(Scalar) * (rowMajor ? (size_t) src.cols() : 1),
                  sizeof(Scalar) * (rowMajor ? 1 : (size_t) src.rows()) }
            )).release();
        }
    }

    PYBIND11_TYPE_CASTER(Type, _("numpy.ndarray[dtype=") + npy_format_descriptor<Scalar>::name() +
            _(", shape=(") + rows() + _(", ") + cols() + _(")]"));

protected:
    template <typename T = Type, typename std::enable_if<T::RowsAtCompileTime == Eigen::Dynamic, int>::type = 0>
    static PYBIND11_DESCR rows() { return _("m"); }
    template <typename T = Type, typename std::enable_if<T::RowsAtCompileTime != Eigen::Dynamic, int>::type = 0>
    static PYBIND11_DESCR rows() { return _<T::RowsAtCompileTime>(); }
    template <typename T = Type, typename std::enable_if<T::ColsAtCompileTime == Eigen::Dynamic, int>::type = 0>
    static PYBIND11_DESCR cols() { return _("n"); }
    template <typename T = Type, typename std::enable_if<T::ColsAtCompileTime != Eigen::Dynamic, int>::type = 0>
    static PYBIND11_DESCR cols() { return _<T::ColsAtCompileTime>(); }
};

template<typename Type>
struct type_caster<Type, typename std::enable_if<is_eigen_sparse<Type>::value>::type> {
    typedef typename Type::Scalar Scalar;
    typedef typename std::remove_reference<decltype(*std::declval<Type>().outerIndexPtr())>::type StorageIndex;
    typedef typename Type::Index Index;
    static constexpr bool rowMajor = Type::Flags & Eigen::RowMajorBit;

    bool load(handle src, bool) {
        if (!src)
            return false;

        object obj(src, true);
        object sparse_module = module::import("scipy.sparse");
        object matrix_type = sparse_module.attr(
            rowMajor ? "csr_matrix" : "csc_matrix");

        if (obj.get_type() != matrix_type.ptr()) {
            try {
                obj = matrix_type(obj);
            } catch (const error_already_set &) {
                PyErr_Clear(); 
                return false;
            }
        }

        auto valuesArray = array_t<Scalar>((object) obj.attr("data"));
        auto innerIndicesArray = array_t<StorageIndex>((object) obj.attr("indices"));
        auto outerIndicesArray = array_t<StorageIndex>((object) obj.attr("indptr"));
        auto shape = pybind11::tuple((pybind11::object) obj.attr("shape"));
        auto nnz = obj.attr("nnz").cast<Index>();

        if (!valuesArray.check() || !innerIndicesArray.check() ||
            !outerIndicesArray.check())
            return false;

        buffer_info outerIndices = outerIndicesArray.request();
        buffer_info innerIndices = innerIndicesArray.request();
        buffer_info values = valuesArray.request();

        value = Eigen::MappedSparseMatrix<Scalar, Type::Flags, StorageIndex>(
            shape[0].cast<Index>(),
            shape[1].cast<Index>(),
            nnz,
            static_cast<StorageIndex *>(outerIndices.ptr),
            static_cast<StorageIndex *>(innerIndices.ptr),
            static_cast<Scalar *>(values.ptr)
        );

        return true;
    }

    static handle cast(const Type &src, return_value_policy /* policy */, handle /* parent */) {
        const_cast<Type&>(src).makeCompressed();

        object matrix_type = module::import("scipy.sparse").attr(
            rowMajor ? "csr_matrix" : "csc_matrix");

        array data(buffer_info(
            // Pointer to buffer
            const_cast<Scalar *>(src.valuePtr()),
            // Size of one scalar
            sizeof(Scalar),
            // Python struct-style format descriptor
            format_descriptor<Scalar>::value,
            // Number of dimensions
            1,
            // Buffer dimensions
            { (size_t) src.nonZeros() },
            // Strides
            { sizeof(Scalar) }
        ));

        array outerIndices(buffer_info(
            // Pointer to buffer
            const_cast<StorageIndex *>(src.outerIndexPtr()),
            // Size of one scalar
            sizeof(StorageIndex),
            // Python struct-style format descriptor
            format_descriptor<StorageIndex>::value,
            // Number of dimensions
            1,
            // Buffer dimensions
            { (size_t) (rowMajor ? src.rows() : src.cols()) + 1 },
            // Strides
            { sizeof(StorageIndex) }
        ));

        array innerIndices(buffer_info(
            // Pointer to buffer
            const_cast<StorageIndex *>(src.innerIndexPtr()),
            // Size of one scalar
            sizeof(StorageIndex),
            // Python struct-style format descriptor
            format_descriptor<StorageIndex>::value,
            // Number of dimensions
            1,
            // Buffer dimensions
            { (size_t) src.nonZeros() },
            // Strides
            { sizeof(StorageIndex) }
        ));

        return matrix_type(
            std::make_tuple(data, innerIndices, outerIndices),
            std::make_pair(src.rows(), src.cols())
        ).release();
    }

    PYBIND11_TYPE_CASTER(Type, _<(Type::Flags & Eigen::RowMajorBit) != 0>("scipy.sparse.csr_matrix[dtype=", "scipy.sparse.csc_matrix[dtype=")
            + npy_format_descriptor<Scalar>::name() + _("]"));
};

NAMESPACE_END(detail)
NAMESPACE_END(pybind11)

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
