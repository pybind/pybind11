/*
    tests/test_buffers.cpp -- supporting Pythons' buffer protocol

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <pybind11/complex.h>
#include <pybind11/stl.h>

#include "constructor_stats.h"
#include "pybind11_tests.h"

TEST_SUBMODULE(buffers, m) {
    m.attr("long_double_and_double_have_same_size") = (sizeof(long double) == sizeof(double));

    m.def("format_descriptor_format_buffer_info_equiv",
          [](const std::string &cpp_name, const py::buffer &buffer) {
              // https://google.github.io/styleguide/cppguide.html#Static_and_Global_Variables
              static auto *format_table = new std::map<std::string, std::string>;
              static auto *equiv_table
                  = new std::map<std::string, bool (py::buffer_info::*)() const>;
              if (format_table->empty()) {
#define PYBIND11_ASSIGN_HELPER(...)                                                               \
    (*format_table)[#__VA_ARGS__] = py::format_descriptor<__VA_ARGS__>::format();                 \
    (*equiv_table)[#__VA_ARGS__] = &py::buffer_info::item_type_is_equivalent_to<__VA_ARGS__>;
                  PYBIND11_ASSIGN_HELPER(PyObject *)
                  PYBIND11_ASSIGN_HELPER(bool)
                  PYBIND11_ASSIGN_HELPER(std::int8_t)
                  PYBIND11_ASSIGN_HELPER(std::uint8_t)
                  PYBIND11_ASSIGN_HELPER(std::int16_t)
                  PYBIND11_ASSIGN_HELPER(std::uint16_t)
                  PYBIND11_ASSIGN_HELPER(std::int32_t)
                  PYBIND11_ASSIGN_HELPER(std::uint32_t)
                  PYBIND11_ASSIGN_HELPER(std::int64_t)
                  PYBIND11_ASSIGN_HELPER(std::uint64_t)
                  PYBIND11_ASSIGN_HELPER(float)
                  PYBIND11_ASSIGN_HELPER(double)
                  PYBIND11_ASSIGN_HELPER(long double)
                  PYBIND11_ASSIGN_HELPER(std::complex<float>)
                  PYBIND11_ASSIGN_HELPER(std::complex<double>)
                  PYBIND11_ASSIGN_HELPER(std::complex<long double>)
#undef PYBIND11_ASSIGN_HELPER
              }
              return std::pair<std::string, bool>(
                  (*format_table)[cpp_name], (buffer.request().*((*equiv_table)[cpp_name]))());
          });

    // test_from_python / test_to_python:
    class Matrix {
    public:
        Matrix(py::ssize_t rows, py::ssize_t cols) : m_rows(rows), m_cols(cols) {
            print_created(this, std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
            // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
            m_data = new float[(size_t) (rows * cols)];
            memset(m_data, 0, sizeof(float) * (size_t) (rows * cols));
        }

        Matrix(const Matrix &s) : m_rows(s.m_rows), m_cols(s.m_cols) {
            print_copy_created(this,
                               std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
            // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
            m_data = new float[(size_t) (m_rows * m_cols)];
            memcpy(m_data, s.m_data, sizeof(float) * (size_t) (m_rows * m_cols));
        }

        Matrix(Matrix &&s) noexcept : m_rows(s.m_rows), m_cols(s.m_cols), m_data(s.m_data) {
            print_move_created(this);
            s.m_rows = 0;
            s.m_cols = 0;
            s.m_data = nullptr;
        }

        ~Matrix() {
            print_destroyed(this,
                            std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
            delete[] m_data;
        }

        Matrix &operator=(const Matrix &s) {
            if (this == &s) {
                return *this;
            }
            print_copy_assigned(this,
                                std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
            delete[] m_data;
            m_rows = s.m_rows;
            m_cols = s.m_cols;
            m_data = new float[(size_t) (m_rows * m_cols)];
            memcpy(m_data, s.m_data, sizeof(float) * (size_t) (m_rows * m_cols));
            return *this;
        }

        Matrix &operator=(Matrix &&s) noexcept {
            print_move_assigned(this,
                                std::to_string(m_rows) + "x" + std::to_string(m_cols) + " matrix");
            if (&s != this) {
                delete[] m_data;
                m_rows = s.m_rows;
                m_cols = s.m_cols;
                m_data = s.m_data;
                s.m_rows = 0;
                s.m_cols = 0;
                s.m_data = nullptr;
            }
            return *this;
        }

        float operator()(py::ssize_t i, py::ssize_t j) const {
            return m_data[(size_t) (i * m_cols + j)];
        }

        float &operator()(py::ssize_t i, py::ssize_t j) {
            return m_data[(size_t) (i * m_cols + j)];
        }

        float *data() { return m_data; }

        py::ssize_t rows() const { return m_rows; }
        py::ssize_t cols() const { return m_cols; }

    private:
        py::ssize_t m_rows;
        py::ssize_t m_cols;
        float *m_data;
    };
    py::class_<Matrix>(m, "Matrix", py::buffer_protocol())
        .def(py::init<py::ssize_t, py::ssize_t>())
        /// Construct from a buffer
        .def(py::init([](const py::buffer &b) {
            py::buffer_info info = b.request();
            if (info.format != py::format_descriptor<float>::format() || info.ndim != 2) {
                throw std::runtime_error("Incompatible buffer format!");
            }

            auto *v = new Matrix(info.shape[0], info.shape[1]);
            memcpy(v->data(), info.ptr, sizeof(float) * (size_t) (v->rows() * v->cols()));
            return v;
        }))

        .def("rows", &Matrix::rows)
        .def("cols", &Matrix::cols)

        /// Bare bones interface
        .def("__getitem__",
             [](const Matrix &m, std::pair<py::ssize_t, py::ssize_t> i) {
                 if (i.first >= m.rows() || i.second >= m.cols()) {
                     throw py::index_error();
                 }
                 return m(i.first, i.second);
             })
        .def("__setitem__",
             [](Matrix &m, std::pair<py::ssize_t, py::ssize_t> i, float v) {
                 if (i.first >= m.rows() || i.second >= m.cols()) {
                     throw py::index_error();
                 }
                 m(i.first, i.second) = v;
             })
        /// Provide buffer access
        .def_buffer([](Matrix &m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),                          /* Pointer to buffer */
                {m.rows(), m.cols()},              /* Buffer dimensions */
                {sizeof(float) * size_t(m.cols()), /* Strides (in bytes) for each index */
                 sizeof(float)});
        });

    // A matrix that uses Fortran storage order.
    class FortranMatrix : public Matrix {
    public:
        FortranMatrix(py::ssize_t rows, py::ssize_t cols) : Matrix(cols, rows) {
            print_created(this,
                          std::to_string(rows) + "x" + std::to_string(cols) + " Fortran matrix");
        }

        float operator()(py::ssize_t i, py::ssize_t j) const { return Matrix::operator()(j, i); }

        float &operator()(py::ssize_t i, py::ssize_t j) { return Matrix::operator()(j, i); }

        using Matrix::data;

        py::ssize_t rows() const { return Matrix::cols(); }
        py::ssize_t cols() const { return Matrix::rows(); }
    };
    py::class_<FortranMatrix, Matrix>(m, "FortranMatrix", py::buffer_protocol())
        .def(py::init<py::ssize_t, py::ssize_t>())

        .def("rows", &FortranMatrix::rows)
        .def("cols", &FortranMatrix::cols)

        /// Bare bones interface
        .def("__getitem__",
             [](const FortranMatrix &m, std::pair<py::ssize_t, py::ssize_t> i) {
                 if (i.first >= m.rows() || i.second >= m.cols()) {
                     throw py::index_error();
                 }
                 return m(i.first, i.second);
             })
        .def("__setitem__",
             [](FortranMatrix &m, std::pair<py::ssize_t, py::ssize_t> i, float v) {
                 if (i.first >= m.rows() || i.second >= m.cols()) {
                     throw py::index_error();
                 }
                 m(i.first, i.second) = v;
             })
        /// Provide buffer access
        .def_buffer([](FortranMatrix &m) -> py::buffer_info {
            return py::buffer_info(m.data(),             /* Pointer to buffer */
                                   {m.rows(), m.cols()}, /* Buffer dimensions */
                                   /* Strides (in bytes) for each index */
                                   {sizeof(float), sizeof(float) * size_t(m.rows())});
        });

    // A matrix that uses a discontiguous underlying memory block.
    class DiscontiguousMatrix : public Matrix {
    public:
        DiscontiguousMatrix(py::ssize_t rows,
                            py::ssize_t cols,
                            py::ssize_t row_factor,
                            py::ssize_t col_factor)
            : Matrix(rows * row_factor, cols * col_factor), m_row_factor(row_factor),
              m_col_factor(col_factor) {
            print_created(this,
                          std::to_string(rows) + "(*" + std::to_string(row_factor) + ")x"
                              + std::to_string(cols) + "(*" + std::to_string(col_factor)
                              + ") matrix");
        }

        ~DiscontiguousMatrix() {
            print_destroyed(this,
                            std::to_string(rows() / m_row_factor) + "(*"
                                + std::to_string(m_row_factor) + ")x"
                                + std::to_string(cols() / m_col_factor) + "(*"
                                + std::to_string(m_col_factor) + ") matrix");
        }

        float operator()(py::ssize_t i, py::ssize_t j) const {
            return Matrix::operator()(i *m_row_factor, j *m_col_factor);
        }

        float &operator()(py::ssize_t i, py::ssize_t j) {
            return Matrix::operator()(i *m_row_factor, j *m_col_factor);
        }

        using Matrix::data;

        py::ssize_t rows() const { return Matrix::rows() / m_row_factor; }
        py::ssize_t cols() const { return Matrix::cols() / m_col_factor; }
        py::ssize_t row_factor() const { return m_row_factor; }
        py::ssize_t col_factor() const { return m_col_factor; }

    private:
        py::ssize_t m_row_factor;
        py::ssize_t m_col_factor;
    };
    py::class_<DiscontiguousMatrix, Matrix>(m, "DiscontiguousMatrix", py::buffer_protocol())
        .def(py::init<py::ssize_t, py::ssize_t, py::ssize_t, py::ssize_t>())

        .def("rows", &DiscontiguousMatrix::rows)
        .def("cols", &DiscontiguousMatrix::cols)

        /// Bare bones interface
        .def("__getitem__",
             [](const DiscontiguousMatrix &m, std::pair<py::ssize_t, py::ssize_t> i) {
                 if (i.first >= m.rows() || i.second >= m.cols()) {
                     throw py::index_error();
                 }
                 return m(i.first, i.second);
             })
        .def("__setitem__",
             [](DiscontiguousMatrix &m, std::pair<py::ssize_t, py::ssize_t> i, float v) {
                 if (i.first >= m.rows() || i.second >= m.cols()) {
                     throw py::index_error();
                 }
                 m(i.first, i.second) = v;
             })
        /// Provide buffer access
        .def_buffer([](DiscontiguousMatrix &m) -> py::buffer_info {
            return py::buffer_info(m.data(),             /* Pointer to buffer */
                                   {m.rows(), m.cols()}, /* Buffer dimensions */
                                   /* Strides (in bytes) for each index */
                                   {size_t(m.col_factor()) * sizeof(float) * size_t(m.cols())
                                        * size_t(m.row_factor()),
                                    size_t(m.col_factor()) * sizeof(float)});
        });

    class BrokenMatrix : public Matrix {
    public:
        BrokenMatrix(py::ssize_t rows, py::ssize_t cols) : Matrix(rows, cols) {}
        void throw_runtime_error() { throw std::runtime_error("See PR #5324 for context."); }
    };
    py::class_<BrokenMatrix>(m, "BrokenMatrix", py::buffer_protocol())
        .def(py::init<py::ssize_t, py::ssize_t>())
        .def_buffer([](BrokenMatrix &m) {
            m.throw_runtime_error();
            return py::buffer_info();
        });

    // test_inherited_protocol
    class SquareMatrix : public Matrix {
    public:
        explicit SquareMatrix(py::ssize_t n) : Matrix(n, n) {}
    };
    // Derived classes inherit the buffer protocol and the buffer access function
    py::class_<SquareMatrix, Matrix>(m, "SquareMatrix").def(py::init<py::ssize_t>());

    // test_pointer_to_member_fn
    // Tests that passing a pointer to member to the base class works in
    // the derived class.
    struct Buffer {
        int32_t value = 0;

        py::buffer_info get_buffer_info() {
            return py::buffer_info(
                &value, sizeof(value), py::format_descriptor<int32_t>::format(), 1);
        }
    };
    py::class_<Buffer>(m, "Buffer", py::buffer_protocol())
        .def(py::init<>())
        .def_readwrite("value", &Buffer::value)
        .def_buffer(&Buffer::get_buffer_info);

    class ConstBuffer {
        std::unique_ptr<int32_t> value;

    public:
        int32_t get_value() const { return *value; }
        void set_value(int32_t v) { *value = v; }

        py::buffer_info get_buffer_info() const {
            return py::buffer_info(
                value.get(), sizeof(*value), py::format_descriptor<int32_t>::format(), 1);
        }

        ConstBuffer() : value(new int32_t{0}) {}
    };
    py::class_<ConstBuffer>(m, "ConstBuffer", py::buffer_protocol())
        .def(py::init<>())
        .def_property("value", &ConstBuffer::get_value, &ConstBuffer::set_value)
        .def_buffer(&ConstBuffer::get_buffer_info);

    struct DerivedBuffer : public Buffer {};
    py::class_<DerivedBuffer>(m, "DerivedBuffer", py::buffer_protocol())
        .def(py::init<>())
        .def_readwrite("value", (int32_t DerivedBuffer::*) &DerivedBuffer::value)
        .def_buffer(&DerivedBuffer::get_buffer_info);

    struct BufferReadOnly {
        const uint8_t value = 0;
        explicit BufferReadOnly(uint8_t value) : value(value) {}

        py::buffer_info get_buffer_info() { return py::buffer_info(&value, 1); }
    };
    py::class_<BufferReadOnly>(m, "BufferReadOnly", py::buffer_protocol())
        .def(py::init<uint8_t>())
        .def_buffer(&BufferReadOnly::get_buffer_info);

    struct BufferReadOnlySelect {
        uint8_t value = 0;
        bool readonly = false;

        py::buffer_info get_buffer_info() { return py::buffer_info(&value, 1, readonly); }
    };
    py::class_<BufferReadOnlySelect>(m, "BufferReadOnlySelect", py::buffer_protocol())
        .def(py::init<>())
        .def_readwrite("value", &BufferReadOnlySelect::value)
        .def_readwrite("readonly", &BufferReadOnlySelect::readonly)
        .def_buffer(&BufferReadOnlySelect::get_buffer_info);

    // Expose buffer_info for testing.
    py::class_<py::buffer_info>(m, "buffer_info")
        .def(py::init<>())
        .def_readonly("itemsize", &py::buffer_info::itemsize)
        .def_readonly("size", &py::buffer_info::size)
        .def_readonly("format", &py::buffer_info::format)
        .def_readonly("ndim", &py::buffer_info::ndim)
        .def_readonly("shape", &py::buffer_info::shape)
        .def_readonly("strides", &py::buffer_info::strides)
        .def_readonly("readonly", &py::buffer_info::readonly)
        .def("__repr__", [](py::handle self) {
            return py::str("itemsize={0.itemsize!r}, size={0.size!r}, format={0.format!r}, "
                           "ndim={0.ndim!r}, shape={0.shape!r}, strides={0.strides!r}, "
                           "readonly={0.readonly!r}")
                .format(self);
        });

    m.def("get_buffer_info", [](const py::buffer &buffer) { return buffer.request(); });

    // Expose Py_buffer for testing.
    m.attr("PyBUF_FORMAT") = PyBUF_FORMAT;
    m.attr("PyBUF_SIMPLE") = PyBUF_SIMPLE;
    m.attr("PyBUF_ND") = PyBUF_ND;
    m.attr("PyBUF_STRIDES") = PyBUF_STRIDES;
    m.attr("PyBUF_INDIRECT") = PyBUF_INDIRECT;
    m.attr("PyBUF_C_CONTIGUOUS") = PyBUF_C_CONTIGUOUS;
    m.attr("PyBUF_F_CONTIGUOUS") = PyBUF_F_CONTIGUOUS;
    m.attr("PyBUF_ANY_CONTIGUOUS") = PyBUF_ANY_CONTIGUOUS;

    m.def("get_py_buffer", [](const py::object &object, int flags) {
        Py_buffer buffer;
        memset(&buffer, 0, sizeof(Py_buffer));
        if (PyObject_GetBuffer(object.ptr(), &buffer, flags) == -1) {
            throw py::error_already_set();
        }

        auto SimpleNamespace = py::module_::import("types").attr("SimpleNamespace");
        py::object result = SimpleNamespace("len"_a = buffer.len,
                                            "readonly"_a = buffer.readonly,
                                            "itemsize"_a = buffer.itemsize,
                                            "format"_a = buffer.format,
                                            "ndim"_a = buffer.ndim,
                                            "shape"_a = py::none(),
                                            "strides"_a = py::none(),
                                            "suboffsets"_a = py::none());
        if (buffer.shape != nullptr) {
            py::list l;
            for (auto i = 0; i < buffer.ndim; i++) {
                l.append(buffer.shape[i]);
            }
            py::setattr(result, "shape", l);
        }
        if (buffer.strides != nullptr) {
            py::list l;
            for (auto i = 0; i < buffer.ndim; i++) {
                l.append(buffer.strides[i]);
            }
            py::setattr(result, "strides", l);
        }
        if (buffer.suboffsets != nullptr) {
            py::list l;
            for (auto i = 0; i < buffer.ndim; i++) {
                l.append(buffer.suboffsets[i]);
            }
            py::setattr(result, "suboffsets", l);
        }

        PyBuffer_Release(&buffer);
        return result;
    });
}
