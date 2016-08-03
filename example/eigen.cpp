/*
    example/eigen.cpp -- automatic conversion of Eigen types

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"
#include <pybind11/eigen.h>
#include <Eigen/Cholesky>

Eigen::VectorXf double_col(const Eigen::VectorXf& x)
{ return 2.0f * x; }

Eigen::RowVectorXf double_row(const Eigen::RowVectorXf& x)
{ return 2.0f * x; }

Eigen::MatrixXf double_mat_cm(const Eigen::MatrixXf& x)
{ return 2.0f * x; }

// Different ways of passing via Eigen::Ref; the first and second are the Eigen-recommended
Eigen::MatrixXd cholesky1(Eigen::Ref<Eigen::MatrixXd> &x) { return x.llt().matrixL(); }
Eigen::MatrixXd cholesky2(const Eigen::Ref<const Eigen::MatrixXd> &x) { return x.llt().matrixL(); }
Eigen::MatrixXd cholesky3(const Eigen::Ref<Eigen::MatrixXd> &x) { return x.llt().matrixL(); }
Eigen::MatrixXd cholesky4(Eigen::Ref<const Eigen::MatrixXd> &x) { return x.llt().matrixL(); }
Eigen::MatrixXd cholesky5(Eigen::Ref<Eigen::MatrixXd> x) { return x.llt().matrixL(); }
Eigen::MatrixXd cholesky6(Eigen::Ref<const Eigen::MatrixXd> x) { return x.llt().matrixL(); }

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXfRowMajor;
MatrixXfRowMajor double_mat_rm(const MatrixXfRowMajor& x)
{ return 2.0f * x; }

void init_eigen(py::module &m) {
    typedef Eigen::Matrix<float, 5, 6, Eigen::RowMajor> FixedMatrixR;
    typedef Eigen::Matrix<float, 5, 6> FixedMatrixC;
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DenseMatrixR;
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> DenseMatrixC;
    typedef Eigen::SparseMatrix<float, Eigen::RowMajor> SparseMatrixR;
    typedef Eigen::SparseMatrix<float> SparseMatrixC;

    // Non-symmetric matrix with zero elements
    Eigen::MatrixXf mat(5, 6);
    mat << 0, 3, 0, 0, 0, 11, 22, 0, 0, 0, 17, 11, 7, 5, 0, 1, 0, 11, 0,
        0, 0, 0, 0, 11, 0, 0, 14, 0, 8, 11;

    m.def("double_col", &double_col);
    m.def("double_row", &double_row);
    m.def("double_mat_cm", &double_mat_cm);
    m.def("double_mat_rm", &double_mat_rm);
    m.def("cholesky1", &cholesky1);
    m.def("cholesky2", &cholesky2);
    m.def("cholesky3", &cholesky3);
    m.def("cholesky4", &cholesky4);
    m.def("cholesky5", &cholesky5);
    m.def("cholesky6", &cholesky6);

    m.def("fixed_r", [mat]() -> FixedMatrixR { 
        return FixedMatrixR(mat);
    });

    m.def("fixed_c", [mat]() -> FixedMatrixC { 
        return FixedMatrixC(mat);
    });

    m.def("fixed_passthrough_r", [](const FixedMatrixR &m) -> FixedMatrixR { 
        return m;
    });

    m.def("fixed_passthrough_c", [](const FixedMatrixC &m) -> FixedMatrixC { 
        return m;
    });

    m.def("dense_r", [mat]() -> DenseMatrixR { 
        return DenseMatrixR(mat);
    });

    m.def("dense_c", [mat]() -> DenseMatrixC { 
        return DenseMatrixC(mat);
    });

    m.def("dense_passthrough_r", [](const DenseMatrixR &m) -> DenseMatrixR { 
        return m;
    });

    m.def("dense_passthrough_c", [](const DenseMatrixC &m) -> DenseMatrixC { 
        return m;
    });

    m.def("sparse_r", [mat]() -> SparseMatrixR { 
        return Eigen::SparseView<Eigen::MatrixXf>(mat);
    });

    m.def("sparse_c", [mat]() -> SparseMatrixC { 
        return Eigen::SparseView<Eigen::MatrixXf>(mat);
    });

    m.def("sparse_passthrough_r", [](const SparseMatrixR &m) -> SparseMatrixR { 
        return m;
    });

    m.def("sparse_passthrough_c", [](const SparseMatrixC &m) -> SparseMatrixC { 
        return m;
    });
}
