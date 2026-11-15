.. _supported:

Supported versions
##################

This page lists the oldest version of each tool or dependency that pybind11
supports. Each entry has two values:

**Minimum expected**
    The oldest version that we intend to work. Usually the headers or the build
    system reject older versions with an error. We fix bugs reported for these
    versions, but CI does not always test them.

**Minimum tested**
    The oldest version that CI tests. Versions between the two minimums
    usually work, but we do not test them.

When you change one of these minimums (in the headers, in CMake, in
``pyproject.toml``, or in the CI matrix), update this table too.

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
     -
   * - PyPy
     - 3.10
     - 3.11 (v7.3.23)
     -
   * - GraalPy
     - 25.0
     - 25.0
     -

We plan to drop 3.13t support soon, as it was experimental. Use 3.14t or newer.

C++ standard and compilers
==========================

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Compiler
     - Minimum expected
     - Minimum tested
     - Notes
   * - C++ standard
     - C++11
     - C++11
     - CI tests C++11 through C++23.
   * - GCC
     - 4.8
     - 8 (AlmaLinux 8)
     -
   * - Clang
     - 3.3
     - 5
     -
   * - Apple Clang
     - Xcode 5
     - Xcode 16
     - The version of the default macOS runner.
   * - MSVC
     - 2019 (19.20)
     - 2022
     - GitHub Actions does not have MSVC 2019 runners.
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
     - The newest tested version is 4.2.
   * - Eigen
     - 3.2.7
     - 3.3.9
     - ``pybind11/eigen/tensor.h`` needs Eigen 3.3.0 or newer.
   * - NumPy
     -
     - 1.21
     - NumPy 2 needs pybind11 2.12 or newer.
   * - scikit-build-core
     - 0.11.2
     - latest
     - Only needed to build the pybind11 Python package from source.
