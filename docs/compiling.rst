Build systems
#############

Building with setuptools
========================

For projects on PyPI, building with setuptools is the way to go. Sylvain Corlay
has kindly provided an example project which shows how to set up everything,
including automatic generation of documentation using Sphinx. Please refer to
the [python_example]_ repository.

.. [python_example] https://github.com/pybind/python_example

Building with cppimport
========================

 cppimport is a small Python import hook that determines whether there is a C++
 source file whose name matches the requested module. If there is, the file is
 compiled as a Python extension using pybind11 and placed in the same folder as
 the C++ source file. Python is then able to find the module and load it.

.. [cppimport] https://github.com/tbenthompson/cppimport

.. _cmake:

Building with CMake
===================

For C++ codebases that have an existing CMake-based build system, a Python
extension module can be created with just a few lines of code:

.. code-block:: cmake

    cmake_minimum_required(VERSION 2.8.12)
    project(example)

    add_subdirectory(pybind11)
    pybind11_add_module(example example.cpp)

This assumes that the pybind11 repository is located in a subdirectory named
:file:`pybind11` and that the code is located in a file named :file:`example.cpp`.
The CMake command ``add_subdirectory`` will import a function with the signature
``pybind11_add_module(<name> source1 [source2 ...])``. It will take care of all
the details needed to build a Python extension module on any platform.

The target Python version can be selected by setting the ``PYBIND11_PYTHON_VERSION``
variable before adding the pybind11 subdirectory. Alternatively, an exact Python
installation can be specified by setting ``PYTHON_EXECUTABLE``.

A working sample project, including a way to invoke CMake from :file:`setup.py` for
PyPI integration, can be found in the [cmake_example]_  repository.

.. [cmake_example] https://github.com/pybind/cmake_example
