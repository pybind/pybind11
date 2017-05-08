/*
    tests/test_buffers.cpp -- supporting Pythons' buffer protocol

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include "constructor_stats.h"

class Matrix {
public:
    Matrix(ssize_t rows, ssize_t cols) : m_rows(rows), m_cols(cols) {
        print_created(this, std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
        m_data = new float[(size_t) (rows*cols)];
        memset(m_data, 0, sizeof(float) * (size_t) (rows * cols));
    }

    Matrix(const Matrix &s) : m_rows(s.m_rows), m_cols(s.m_cols) {
        print_copy_created(this, std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
        m_data = new float[(size_t) (m_rows * m_cols)];
        memcpy(m_data, s.m_data, sizeof(float) * (size_t) (m_rows * m_cols));
    }

    Matrix(Matrix &&s) : m_rows(s.m_rows), m_cols(s.m_cols), m_data(s.m_data) {
        print_move_created(this);
        s.m_rows = 0;
        s.m_cols = 0;
        s.m_data = nullptr;
    }

    ~Matrix() {
        print_destroyed(this, std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
        delete[] m_data;
    }

    Matrix &operator=(const Matrix &s) {
        print_copy_assigned(this, std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
        delete[] m_data;
        m_rows = s.m_rows;
        m_cols = s.m_cols;
        m_data = new float[(size_t) (m_rows * m_cols)];
        memcpy(m_data, s.m_data, sizeof(float) * (size_t) (m_rows * m_cols));
        return *this;
    }

    Matrix &operator=(Matrix &&s) {
        print_move_assigned(this, std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
        if (&s != this) {
            delete[] m_data;
            m_rows = s.m_rows; m_cols = s.m_cols; m_data = s.m_data;
            s.m_rows = 0; s.m_cols = 0; s.m_data = nullptr;
        }
        return *this;
    }

    float operator()(ssize_t i, ssize_t j) const {
        return m_data[(size_t) (i*m_cols + j)];
    }

    float &operator()(ssize_t i, ssize_t j) {
        return m_data[(size_t) (i*m_cols + j)];
    }

    float *data() { return m_data; }

    ssize_t rows() const { return m_rows; }
    ssize_t cols() const { return m_cols; }
private:
    ssize_t m_rows;
    ssize_t m_cols;
    float *m_data;
};

test_initializer buffers([](py::module &m) {
    py::class_<Matrix> mtx(m, "Matrix", py::buffer_protocol());

    mtx.def(py::init<ssize_t, ssize_t>())
        /// Construct from a buffer
        .def("__init__", [](Matrix &v, py::buffer b) {
            py::buffer_info info = b.request();
            if (info.format != py::format_descriptor<float>::format() || info.ndim != 2)
                throw std::runtime_error("Incompatible buffer format!");
            new (&v) Matrix(info.shape[0], info.shape[1]);
            memcpy(v.data(), info.ptr, sizeof(float) * (size_t) (v.rows() * v.cols()));
        })

       .def("rows", &Matrix::rows)
       .def("cols", &Matrix::cols)

        /// Bare bones interface
       .def("__getitem__", [](const Matrix &m, std::pair<ssize_t, ssize_t> i) {
            if (i.first >= m.rows() || i.second >= m.cols())
                throw py::index_error();
            return m(i.first, i.second);
        })
       .def("__setitem__", [](Matrix &m, std::pair<ssize_t, ssize_t> i, float v) {
            if (i.first >= m.rows() || i.second >= m.cols())
                throw py::index_error();
            m(i.first, i.second) = v;
        })
       /// Provide buffer access
       .def_buffer([](Matrix &m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),                               /* Pointer to buffer */
                sizeof(float),                          /* Size of one scalar */
                py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
                2,                                      /* Number of dimensions */
                { m.rows(), m.cols() },                 /* Buffer dimensions */
                { sizeof(float) * size_t(m.rows()),     /* Strides (in bytes) for each index */
                  sizeof(float) }
            );
        })
        ;
});
