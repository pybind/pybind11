.. _advanced:

Advanced topics
###############

Operator overloading
====================

Overriding virtual functions in Python
======================================

Passing anonymous functions
===========================

Return value policies
=====================

Functions taking Python objects as arguments
============================================

Callbacks
=========

Buffer protocol
===============

Python supports an extremely general and convenient approach for exchanging
data between plugin libraries. Types can expose a buffer view which provides
fast direct access to the raw internal representation. Suppose we want to bind
the following simplistic Matrix class:

.. code-block:: cpp

    class Matrix {
    public:
        Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols) {
            m_data = new float[rows*cols];
        }
        float *data() { return m_data; }
        size_t rows() const { return m_rows; }
        size_t cols() const { return m_cols; }
    private:
        size_t m_rows, m_cols;
        float *m_data;
    };

The following binding code exposes the ``Matrix`` contents as a buffer object,
making it possible to cast Matrixes into NumPy arrays. It is even possible to
completely avoid copy operations with Python expressions like
``np.array(matrix_instance, copy = False)``.

.. code-block:: cpp

    py::class_<Matrix>(m, "Matrix")
       .def_buffer([](Matrix &m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),                              /* Pointer to buffer */
                sizeof(float),                         /* Size of one scalar */
                py::format_descriptor<float>::value(), /* Python struct-style format descriptor */
                2,                                     /* Number of dimensions */
                { m.rows(), m.cols() },                /* Buffer dimensions */
                { sizeof(float) * m.rows(),            /* Strides (in bytes) for each index */
                  sizeof(float) }
            );
        });

The snippet above binds a lambda function, which can create ``py::buffer_info``
description records on demand describing a given matrix. The contents of
``py::buffer_info`` mirror the Python buffer protocol specification.

.. code-block:: cpp

    struct buffer_info {
        void *ptr;
        size_t itemsize;
        std::string format;
        int ndim;
        std::vector<size_t> shape;
        std::vector<size_t> strides;
    };

To create a C++ function that can take a Python buffer object as an argument,
simply use the type ``py::buffer`` as one of its arguments. Buffers can exist
in a great variety of configurations, hence some safety checks are usually
necessary in the function body. Below, you can see an basic example on how to
define a custom constructor for the Eigen double precision matrix
(``Eigen::MatrixXd``) type, which supports initialization from compatible
buffer
objects (e.g. a NumPy matrix).

.. code-block:: cpp

    py::class_<Eigen::MatrixXd>(m, "MatrixXd")
        .def("__init__", [](Eigen::MatrixXd &m, py::buffer b) {
            /* Request a buffer descriptor from Python */
            py::buffer_info info = b.request();

            /* Some sanity checks ... */
            if (info.format != py::format_descriptor<double>::value())
                throw std::runtime_error("Incompatible format: expected a double array!");

            if (info.ndim != 2)
                throw std::runtime_error("Incompatible buffer dimension!");

            if (info.strides[0] == sizeof(double)) {
                /* Buffer has the right layout -- directly copy. */
                new (&m) Eigen::MatrixXd(info.shape[0], info.shape[1]);
                memcpy(m.data(), info.ptr, sizeof(double) * m.size());
            } else {
                /* Oops -- the buffer is transposed */
                new (&m) Eigen::MatrixXd(info.shape[1], info.shape[0]);
                memcpy(m.data(), info.ptr, sizeof(double) * m.size());
                m.transposeInPlace();
            }
        });

NumPy support
=============

By exchanging ``py::buffer`` with ``py::array`` in the above snippet, we can
restrict the function so that it only accepts NumPy arrays (rather than any
type of Python object satisfying the buffer object protocol).

In many situations, we want to define a function which only accepts a NumPy
array of a certain data type. This is possible via the ``py::array_dtype<T>``
template. For instance, the following function requires the argument to be a
dense array of doubles in C-style ordering.

.. code-block:: cpp

    void f(py::array_dtype<double> array);

When it is invoked with a different type (e.g. an integer), the binding code
will attempt to cast the input into a NumPy array of the requested type.
Note that this feature requires the ``pybind/numpy.h`` header to be included.

Vectorizing functions
=====================

Suppose we want to bind a function with the following signature to Python so
that it can process arbitrary NumPy array arguments (vectors, matrices, general
N-D arrays) in addition to its normal arguments:

.. code-block:: cpp

    double my_func(int x, float y, double z);

After including the ``pybind/numpy.h`` header, this is extremely simple:

.. code-block:: cpp

    m.def("vectorized_func", py::vectorize(my_func));

Invoking the function like below causes 4 calls to be made to ``my_func`` with
each of the the array elements. The result is returned as a NumPy array of type
``numpy.dtype.float64``.

.. code-block:: python

    >>> x = np.array([[1, 3],[5, 7]])
    >>> y = np.array([[2, 4],[6, 8]])
    >>> z = 3
    >>> result = vectorized_func(x, y, z)

The scalar argument ``z`` is transparently replicated 4 times.  The input
arrays ``x`` and ``y`` are automatically converted into the right types (they
are of type  ``numpy.dtype.int64`` but need to be ``numpy.dtype.int32`` and
``numpy.dtype.float32``, respectively)

Sometimes we might want to explitly exclude an argument from the vectorization
because it makes little sense to wrap it in a NumPy array. For instance,
suppose the function signature was

.. code-block:: cpp

    double my_func(int x, float y, my_custom_type *z);

This can be done with a stateful Lambda closure:

.. code-block:: cpp

    // Vectorize a lambda function with a capture object (e.g. to exclude some arguments from the vectorization)
    m.def("vectorized_func",
        [](py::array_dtype<int> x, py::array_dtype<float> y, my_custom_type *z) {
            auto stateful_closure = [z](int x, float y) { return my_func(x, y, z); };
            return py::vectorize(stateful_closure)(x, y);
        }
    );

Throwing exceptions
===================

STL data structures
===================

Smart pointers
==============

.. _custom_constructors:

Custom constructors
===================
