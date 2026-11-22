.. _supported:

Supported versions
##################

This page lists the oldest version of each tool or dependency that pybind11
supports. Each entry has two values:

**Minimum tested**
    The oldest version that CI tests.

**Minimum expected**
    The oldest version that we hope works. Usually the headers or the build
    system reject older versions with an error. We fix bugs reported for this
    version and later, but CI does not test them.

Python
======

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Implementation
     - Minimum expected
     - Minimum tested
     - Notes
   * - CPython
     - 3.9
     - 3.9
     -
   * - CPython, free-threaded
     - 3.13t
     - 3.14t
     - See :ref:`misc_free_threading`.
   * - CPython, stable ABI (``abi3``)
     - 3.12
     - 3.12
     - See :doc:`advanced/stable_abi`.
   * - CPython, free-threaded stable ABI (``abi3t``)
     - 3.15t
     - 3.15t
     - First release of ``abi3t``.
   * - PyPy
     - 3.10 (v7.3.17)
     - 3.11 (v7.3.23)
     - We support v8 too.
   * - GraalPy
     - 25.0
     - 25.0
     - CI tested 24.x before pybind11 3.1. It can work, but we do not test it.

We plan to drop 3.13t support soon, as it was experimental. Use 3.14t or newer.

C++ standard and compilers
==========================

We support C++11 through C++23.

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Compiler
     - Minimum expected
     - Minimum tested
     - Notes
   * - GCC
     -
     - 8 (AlmaLinux 8)
     - 4.8 is the historical cutoff in the headers.
   * - Clang
     - 3.3
     - 5
     -
   * - Apple Clang
     - Xcode 5
     - Xcode 16
     - The default Xcode on the ``macos-15-intel`` runner.
   * - MSVC
     - 2019 (19.20)
     - 2019 (19.29)
     - Tested on AppVeyor.
   * - Intel ICC (classic)
     - 18 (C++11), 19 (C++14)
     - Latest classic ICC
     -
   * - NVCC (CUDA)
     -
     - 12.2
     -
   * - NVIDIA HPC SDK
     -
     - 26.5
     -

Build tools and optional dependencies
=====================================

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Tool
     - Minimum expected
     - Minimum tested
     - Notes
   * - CMake
     - 3.15
     - 3.15
     - 4.x supported too.
   * - Eigen
     - 3.2.7
     - 3.3.7
     - ``pybind11/eigen/tensor.h`` needs Eigen 3.3.0 or newer.
   * - NumPy
     -
     - 1.19.5
     - NumPy 2 needs pybind11 2.12 or newer.
   * - scikit-build-core
     - 0.11.2
     - latest
     - Only needed to build the pybind11 Python package from source.
