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

For CMake-based projects that don't include the pybind11
repository internally, an external installation can be detected
through `find_package(pybind11 ... CONFIG ...)`. See the `Config file
<https://github.com/pybind/pybind11/blob/master/tools/pybind11Config.cmake.in>`_
docstring for details of relevant CMake variables.

Once detected, and after setting any variables to guide Python and
C++ standard detection, the aforementioned ``pybind11_add_module``
wrapper to ``add_library`` can be employed as described above (after
``include(pybind11Tools)``). This procedure is available when using CMake
>= 2.8.12. A working example can be found at [test_installed_module]_ .

.. code-block:: cmake

    cmake_minimum_required(VERSION 2.8.12)
    project(example)

    find_package(pybind11 REQUIRED)
    pybind11_add_module(example example.cpp)

.. [test_installed_module] https://github.com/pybind/pybind11/blob/master/tests/test_installed_module/CMakeLists.txt

When using a version of CMake greater than 3.0, pybind11 can
additionally be used as a special *interface library* following the
call to ``find_package``. CMake variables to guide Python and C++
standard detection should be set *before* ``find_package``. When
``find_package`` returns, the target ``pybind11::pybind11`` is
available with pybind11 headers, Python headers and libraries as
needed, and C++ compile definitions attached. This target is suitable
for linking to an independently constructed (through ``add_library``,
not ``pybind11_add_module``) target in the consuming project. A working
example can be found at [test_installed_target]_ .

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.0)
    project(example)

    add_library(example MODULE main.cpp)

    find_package(pybind11 REQUIRED)
    target_link_libraries(example PRIVATE pybind11::pybind11)
    set_target_properties(example PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                             SUFFIX "${PYTHON_MODULE_EXTENSION}")

.. warning::

    Since pybind11 is a metatemplate library, it is crucial that certain
    compiler flags are provided to ensure high quality code generation. In
    contrast to the ``pybind11_add_module()`` command, the CMake interface
    library only provides the *minimal* set of parameters to ensure that the
    code using pybind11 compiles, but it does **not** pass these extra compiler
    flags (i.e. this is up to you).

    These include Link Time Optimization (``-flto`` on GCC/Clang/ICPC, ``/GL``
    and ``/LTCG`` on Visual Studio). Default-hidden symbols on GCC/Clang/ICPC
    (``-fvisibility=hidden``) and .OBJ files with many sections on Visual Studio
    (``/bigobj``). The :ref:`FAQ <faq:symhidden>` contains an
    explanation on why these are needed.

.. [test_installed_target] https://github.com/pybind/pybind11/blob/master/tests/test_installed_target/CMakeLists.txt

