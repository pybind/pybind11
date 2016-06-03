.. _basics:

First steps
###########

This sections demonstrates the basic features of pybind11. Before getting
started, make sure that development environment is set up to compile the
included set of examples, which also double as test cases.


Compiling the test cases
========================

Linux/MacOS
-----------

On Linux  you'll need to install the **python-dev** or **python3-dev** packages as
well as **cmake**. On Mac OS, the included python version works out of the box,
but **cmake** must still be installed.

After installing the prerequisites, run

.. code-block:: bash

   cmake .
   make -j 4

followed by

.. code-block:: bash

   make test

Windows
-------

On Windows, use the `CMake GUI`_ to create a Visual Studio project. Note that
only the 2015 release and newer versions are supported since pybind11 relies on
various C++11 language features that break older versions of Visual Studio.
After running CMake, open the created :file:`pybind11.sln` file and perform a
release build, which will will produce a file named
:file:`Release\\example.pyd`. Copy this file to the :file:`example` directory
and run :file:`example\\run_test.py` using the targeted Python version.

.. _`CMake GUI`: https://cmake.org/runningcmake

.. Note::

    When all tests fail, make sure that

    1. The Python binary and the testcases are compiled for the same processor
       type and bitness (i.e. either **i386** or **x86_64**)

    2. The Python binary used to run :file:`example\\run_test.py` matches the
       Python version specified in the CMake GUI. This is controlled via
       the ``PYTHON_EXECUTABLE`` ``PYTHON_INCLUDE_DIR``, and
       ``PYTHON_LIBRARY`` variables.

.. seealso::

    Advanced users who are already familiar with Boost.Python may want to skip
    the tutorial and look at the test cases in the :file:`example` directory,
    which exercise all features of pybind11.

Creating bindings for a simple function
=======================================

Let's start by creating Python bindings for an extremely simple function, which
adds two numbers and returns their result:

.. code-block:: cpp

    int add(int i, int j) {
        return i + j;
    }

For simplicity [#f1]_, we'll put both this function and the binding code into
a file named :file:`example.cpp` with the following contents:

.. code-block:: cpp

    #include <pybind11/pybind11.h>

    int add(int i, int j) {
        return i + j;
    }

    namespace py = pybind11;

    PYBIND11_PLUGIN(example) {
        py::module m("example", "pybind11 example plugin");

        m.def("add", &add, "A function which adds two numbers");

        return m.ptr();
    }

The :func:`PYBIND11_PLUGIN` macro creates a function that will be called when an
``import`` statement is issued from within Python. The next line creates a
module named ``example`` (with the supplied docstring). The method
:func:`module::def` generates binding code that exposes the
``add()`` function to Python. The last line returns the internal Python object
associated with ``m`` to the Python interpreter.

.. note::

    Notice how little code was needed to expose our function to Python: all
    details regarding the function's parameters and return value were
    automatically inferred using template metaprogramming. This overall
    approach and the used syntax are borrowed from Boost.Python, though the
    underlying implementation is very different.

pybind11 is a header-only-library, hence it is not necessary to link against
any special libraries (other than Python itself). On Windows, use the CMake
build file discussed in section :ref:`cmake`. On Linux and Mac OS, the above
example can be compiled using the following command

.. code-block:: bash

    $ c++ -O3 -shared -std=c++11 -I <path-to-pybind11>/include `python-config --cflags --ldflags` example.cpp -o example.so

In general, it is advisable to include several additional build parameters
that can considerably reduce the size of the created binary. Refer to section
:ref:`cmake` for a detailed example of a suitable cross-platform CMake-based
build system.

Assuming that the created file :file:`example.so` (:file:`example.pyd` on Windows)
is located in the current directory, the following interactive Python session
shows how to load and execute the example.

.. code-block:: pycon

    $ python
    Python 2.7.10 (default, Aug 22 2015, 20:33:39)
    [GCC 4.2.1 Compatible Apple LLVM 7.0.0 (clang-700.0.59.1)] on darwin
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import example
    >>> example.add(1, 2)
    3L
    >>>

.. _keyword_args:

Keyword arguments
=================

With a simple modification code, it is possible to inform Python about the
names of the arguments ("i" and "j" in this case).

.. code-block:: cpp

    m.def("add", &add, "A function which adds two numbers",
          py::arg("i"), py::arg("j"));

:class:`arg` is one of several special tag classes which can be used to pass
metadata into :func:`module::def`. With this modified binding code, we can now
call the function using keyword arguments, which is a more readable alternative
particularly for functions taking many parameters:

.. code-block:: pycon

    >>> import example
    >>> example.add(i=1, j=2)
    3L

The keyword names also appear in the function signatures within the documentation.

.. code-block:: pycon

    >>> help(example)

    ....

    FUNCTIONS
        add(...)
            Signature : (i: int, j: int) -> int

            A function which adds two numbers

A shorter notation for named arguments is also available:

.. code-block:: cpp
    
    // regular notation
    m.def("add1", &add, py::arg("i"), py::arg("j"));
    // shorthand
    using namespace pybind11::literals;
    m.def("add2", &add, "i"_a, "j"_a);

The :var:`_a` suffix forms a C++11 literal which is equivalent to :class:`arg`. 
Note that the literal operator must first be made visible with the directive 
``using namespace pybind11::literals``. This does not bring in anything else 
from the ``pybind11`` namespace except for literals.

.. _default_args:

Default arguments
=================

Suppose now that the function to be bound has default arguments, e.g.:

.. code-block:: cpp

    int add(int i = 1, int j = 2) {
        return i + j;
    }

Unfortunately, pybind11 cannot automatically extract these parameters, since they
are not part of the function's type information. However, they are simple to specify
using an extension of :class:`arg`:

.. code-block:: cpp

    m.def("add", &add, "A function which adds two numbers",
          py::arg("i") = 1, py::arg("j") = 2);

The default values also appear within the documentation.

.. code-block:: pycon

    >>> help(example)

    ....

    FUNCTIONS
        add(...)
            Signature : (i: int = 1, j: int = 2) -> int

            A function which adds two numbers

The shorthand notation is also available for default arguments:

.. code-block:: cpp
    
    // regular notation
    m.def("add1", &add, py::arg("i") = 1, py::arg("j") = 2);
    // shorthand
    m.def("add2", &add, "i"_a=1, "j"_a=2);

.. _supported_types:

Supported data types
====================

The following basic data types are supported out of the box (some may require
an additional extension header to be included). To pass other data structures
as arguments and return values, refer to the section on binding :ref:`classes`.

+---------------------------------+--------------------------+-------------------------------+
|  Data type                      |  Description             | Header file                   |
+=================================+==========================+===============================+
| ``int8_t``, ``uint8_t``         | 8-bit integers           | :file:`pybind11/pybind11.h`   |
+---------------------------------+--------------------------+-------------------------------+
| ``int16_t``, ``uint16_t``       | 16-bit integers          | :file:`pybind11/pybind11.h`   |
+---------------------------------+--------------------------+-------------------------------+
| ``int32_t``, ``uint32_t``       | 32-bit integers          | :file:`pybind11/pybind11.h`   |
+---------------------------------+--------------------------+-------------------------------+
| ``int64_t``, ``uint64_t``       | 64-bit integers          | :file:`pybind11/pybind11.h`   |
+---------------------------------+--------------------------+-------------------------------+
| ``ssize_t``, ``size_t``         | Platform-dependent size  | :file:`pybind11/pybind11.h`   |
+---------------------------------+--------------------------+-------------------------------+
| ``float``, ``double``           | Floating point types     | :file:`pybind11/pybind11.h`   |
+---------------------------------+--------------------------+-------------------------------+
| ``bool``                        | Two-state Boolean type   | :file:`pybind11/pybind11.h`   |
+---------------------------------+--------------------------+-------------------------------+
| ``char``                        | Character literal        | :file:`pybind11/pybind11.h`   |
+---------------------------------+--------------------------+-------------------------------+
| ``wchar_t``                     | Wide character literal   | :file:`pybind11/pybind11.h`   |
+---------------------------------+--------------------------+-------------------------------+
| ``const char *``                | UTF-8 string literal     | :file:`pybind11/pybind11.h`   |
+---------------------------------+--------------------------+-------------------------------+
| ``const wchar_t *``             | Wide string literal      | :file:`pybind11/pybind11.h`   |
+---------------------------------+--------------------------+-------------------------------+
| ``std::string``                 | STL dynamic UTF-8 string | :file:`pybind11/pybind11.h`   |
+---------------------------------+--------------------------+-------------------------------+
| ``std::wstring``                | STL dynamic wide string  | :file:`pybind11/pybind11.h`   |
+---------------------------------+--------------------------+-------------------------------+
| ``std::pair<T1, T2>``           | Pair of two custom types | :file:`pybind11/pybind11.h`   |
+---------------------------------+--------------------------+-------------------------------+
| ``std::tuple<...>``             | Arbitrary tuple of types | :file:`pybind11/pybind11.h`   |
+---------------------------------+--------------------------+-------------------------------+
| ``std::reference_wrapper<...>`` | Reference type wrapper   | :file:`pybind11/pybind11.h`   |
+---------------------------------+--------------------------+-------------------------------+
| ``std::complex<T>``             | Complex numbers          | :file:`pybind11/complex.h`    |
+---------------------------------+--------------------------+-------------------------------+
| ``std::array<T, Size>``         | STL static array         | :file:`pybind11/stl.h`        |
+---------------------------------+--------------------------+-------------------------------+
| ``std::vector<T>``              | STL dynamic array        | :file:`pybind11/stl.h`        |
+---------------------------------+--------------------------+-------------------------------+
| ``std::list<T>``                | STL linked list          | :file:`pybind11/stl.h`        |
+---------------------------------+--------------------------+-------------------------------+
| ``std::map<T1, T2>``            | STL ordered map          | :file:`pybind11/stl.h`        |
+---------------------------------+--------------------------+-------------------------------+
| ``std::unordered_map<T1, T2>``  | STL unordered map        | :file:`pybind11/stl.h`        |
+---------------------------------+--------------------------+-------------------------------+
| ``std::set<T>``                 | STL ordered set          | :file:`pybind11/stl.h`        |
+---------------------------------+--------------------------+-------------------------------+
| ``std::unordered_set<T>``       | STL unordered set        | :file:`pybind11/stl.h`        |
+---------------------------------+--------------------------+-------------------------------+
| ``std::function<...>``          | STL polymorphic function | :file:`pybind11/functional.h` |
+---------------------------------+--------------------------+-------------------------------+
| ``Eigen::Matrix<...>``          | Dense Eigen matrices     | :file:`pybind11/eigen.h`      |
+---------------------------------+--------------------------+-------------------------------+
| ``Eigen::SparseMatrix<...>``    | Sparse Eigen matrices    | :file:`pybind11/eigen.h`      |
+---------------------------------+--------------------------+-------------------------------+


.. [#f1] In practice, implementation and binding code will generally be located
         in separate files.
