![pybind11 logo](https://github.com/wjakob/pybind11/raw/master/logo.png)

# pybind11 — Seamless operability between C++11 and Python

[![Build Status](https://travis-ci.org/wjakob/pybind11.svg?branch=master)](https://travis-ci.org/wjakob/pybind11)

**pybind11** is a lightweight header library that exposes C++ types in Python
and vice versa, mainly to create Python bindings of existing C++ code. Its
goals and syntax are similar to the excellent
[Boost.Python](http://www.boost.org/doc/libs/1_58_0/libs/python/doc/) library
by David Abrahams: to minimize boilerplate code in traditional extension
modules by inferring type information using compile-time introspection.

The main issue with Boost.Python—and the reason for creating such a similar
project—is Boost. Boost is an enormously large and complex suite of utility
libraries that works with almost every C++ compiler in existence. This
compatibility has its cost: arcane template tricks and workarounds are
necessary to support the oldest and buggiest of compiler specimens. Now that
C++11-compatible compilers are widely available, this heavy machinery has
become an excessively large and unnecessary dependency.

Think of this library as a tiny self-contained version of Boost.Python with
everything stripped away that isn't relevant for binding generation. The whole
codebase requires less than 3000 lines of code and only depends on Python (2.7
or 3.x) and the C++ standard library. This compact implementation was possible
thanks to some of the new C++11 language features (tuples, lambda functions and
variadic templates).

## Core features
The following core C++ features can be mapped to Python

- Functions accepting and returning custom data structures per value, reference, or pointer
- Instance methods and static methods
- Overloaded functions
- Instance attributes and static attributes
- Exceptions
- Enumerations
- Callbacks
- Custom operators
- STL data structures
- Smart pointers with reference counting like `std::shared_ptr`
- Internal references with correct reference counting
- C++ classes with virtual (and pure virtual) methods can be extended in Python

## Goodies
In addition to the core functionality, pybind11 provides some extra goodies:

- It's easy to expose the internal storage of custom data types through
  Pythons' buffer protocols. This is handy e.g. for fast conversion between
  C++ matrix classes like Eigen and NumPy without expensive copy operations.

- pybind11 can automatically vectorize functions so that they are transparently
  applied to all entries of one or more NumPy array arguments.

- Python's slice-based access and assignment operations can be supported with
  just a few lines of code.

- pybind11 uses C++11 move constructors and move assignment operators whenever
  possible to efficiently transfer custom data types.

- It is possible to bind C++11 lambda functions with captured variables. The
  lambda capture data is stored inside the resulting Python function object.

## What does the binding code look like?
Here is a simple example. The directory `example` contains many more.
```C++
#include <pybind/pybind.h>
#include <pybind/operators.h>

namespace py = pybind;

/// Example C++ class which should be bound to Python
class Test {
public:
    Test();
    Test(int value);
    std::string toString();
    Test operator+(const Test &e) const;

    void print_dict(py::dict dict) {
        /* Easily interact with Python types */
        for (auto item : dict)
            std::cout << "key=" << item.first << ", "
                      << "value=" << item.second << std::endl;
    }

    int value = 0;
};


PYTHON_PLUGIN(example) {
    py::module m("example", "pybind example plugin");

    py::class_<Test>(m, "Test", "docstring for the Test class")
        .def(py::init<>(), "docstring for constructor 1")
        .def(py::init<int>(), "docstring for constructor 2")
        .def(py::self + py::self, "Addition operator")
        .def("__str__", &Test::toString, "Convert to a string representation")
        .def("print_dict", &Test::print_dict, "Print a Python dictionary")
        .def_readwrite("value", &Test::value, "An instance attribute");

    return m.ptr();
}
```

## A collection of specific use cases (mostly buffer-related for now)
For brevity, let's set
```C++
namespace py = pybind;
```
### Exposing buffer views
Python supports an extremely general and convenient approach for exchanging
data between plugin libraries. Types can expose a buffer view which provides
fast direct access to the raw internal representation. Suppose we want to bind
the following simplistic Matrix class:

```C++
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
```
The following binding code exposes the ``Matrix`` contents as a buffer object,
making it possible to cast Matrixes into NumPy arrays. It is even possible to
completely avoid copy operations with Python expressions like
``np.array(matrix_instance, copy = False)``.
```C++
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
```
The snippet above binds a lambda function, which can create ``py::buffer_info``
description records on demand describing a given matrix. The contents of
``py::buffer_info``  mirror the Python buffer protocol specification.
```C++
struct buffer_info {
    void *ptr;
    size_t itemsize;
    std::string format;
    int ndim;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
};
```
### Taking Python buffer objects as arguments
To create a C++ function that can take a Python buffer object as an argument,
simply use the type ``py::buffer`` as one of its arguments. Buffers can exist
in a great variety of configurations, hence some safety checks are usually
necessary in the function body. Below, you can see an basic example on how to
define a custom constructor for the Eigen double precision matrix
(``Eigen::MatrixXd``) type, which supports initialization from compatible
buffer
objects (e.g. a NumPy matrix).
```C++
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
```

### Taking NumPy arrays as arguments

By exchanging ``py::buffer`` with ``py::array`` in the above snippet, we can
restrict the function so that it only accepts NumPy arrays (rather than any
type of Python object satisfying the buffer object protocol).

In many situations, we want to define a function which only accepts a NumPy
array of a certain data type. This is possible via the ``py::array_dtype<T>``
template. For instance, the following function requires the argument to be a
dense array of doubles in C-style ordering.
```C++
void f(py::array_dtype<double> array);
```
When it is invoked with a different type (e.g. an integer), the binding code
will attempt to cast the input into a NumPy array of the requested type.

### Auto-vectorizing a function over NumPy array arguments
Suppose we want to bind a function with the following signature to Python so
that it can process arbitrary NumPy array arguments (vectors, matrices, general
N-D arrays) in addition to its normal arguments:
```C++
double my_func(int x, float y, double z);
```
This is extremely simple to do!
```C++
m.def("vectorized_func", py::vectorize(my_func));
```
Invoking the function like below causes 4 calls to be made to ``my_func`` with
each of the the array elements. The result is returned as a NumPy array of type
``numpy.dtype.float64``.
```Python
>>> x = np.array([[1, 3],[5, 7]])
>>> y = np.array([[2, 4],[6, 8]])
>>> z = 3
>>> result = vectorized_func(x, y, z)
```
The scalar argument ``z`` is transparently replicated 4 times.  The input
arrays ``x`` and ``y`` are automatically converted into the right types (they
are of type  ``numpy.dtype.int64`` but need to be ``numpy.dtype.int32`` and
``numpy.dtype.float32``, respectively)

Sometimes we might want to explitly exclude an argument from the vectorization
because it makes little sense to wrap it in a NumPy array. For instance,
suppose the function signature was
```C++
double my_func(int x, float y, my_custom_type *z);
```
This can be done with a stateful Lambda closure:
```C++
// Vectorize a lambda function with a capture object (e.g. to exclude some arguments from the vectorization)
m.def("vectorized_func",
    [](py::array_dtype<int> x, py::array_dtype<float> y, my_custom_type *z) {
        auto stateful_closure = [z](int x, float y) { return my_func(x, y, z); };
        return py::vectorize(stateful_closure)(x, y);
    }
);
```
