# pybind11 -- Seamless operability between C++11 and Python

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
codebase requires less than 2000 lines of code and just depends on Python and
the C++ standard library. This compact implementation was possible thanks to
some of the new C++11 language features (tuples, lambda functions and variadic
templates), and by only targeting Python 3.x and higher.

## Core features
The following C++ features can be mapped to Python

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

## Goodies
In addition to the core functionality, pybind11 provides some extra goodies:

- It's easy to expose the internal storage of custom data types through
  Pythons' buffer protocols. This is handy e.g. for fast conversion between
  C++ matrix classes like Eigen and NumPy without expensive copy operations.

- Python's slice-based access and assignment operations can be supported with
  just a few lines of code.

- pybind11 uses C++11 move constructors and move assignment operators whenever
  possible to efficiently transfer custom data types.

## Limitations
Various things that Boost.Python can do remain unsupported, e.g.:

- Fine grained exception translation: currently, all exceptions derived from
  `std::exception` are mapped to a Python `Exception`, but that's it.

- Default arguments in C++ functions are ignored, though their effect can be
  emulated by binding multiple overloads using anonymous functions.

- Python keyword arguments are not supported in bindings

- Weak pointers are not supported

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
