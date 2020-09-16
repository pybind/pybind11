.. _compiling:

Build systems
#############

.. _build-setuptools:

Building with setuptools
========================

For projects on PyPI, building with setuptools is the way to go. Sylvain Corlay
has kindly provided an example project which shows how to set up everything,
including automatic generation of documentation using Sphinx. Please refer to
the [python_example]_ repository.

.. [python_example] https://github.com/pybind/python_example

A helper file is provided with pybind11 that can simplify usage with setuptools.

To use pybind11 inside your ``setup.py``, you have to have some system to
ensure that ``pybind11`` is installed when you build your package. There are
four possible ways to do this, and pybind11 supports all four: You can ask all
users to install pybind11 beforehand (bad), you can use
:ref:`setup_helpers-pep518` (good, but very new and requires Pip 10),
:ref:`setup_helpers-setup_requires` (discouraged by Python packagers now that
PEP 518 is available, but it still works everywhere), or you can
:ref:`setup_helpers-copy-manually` (always works but you have to manually sync
your copy to get updates).

An example of a ``setup.py`` using pybind11's helpers:

.. code-block:: python

    from setuptools import setup
    from pybind11.setup_helpers import Pybind11Extension

    ext_modules = [
        Pybind11Extension(
            "python_example",
            ["src/main.cpp"],
        ),
    ]

    setup(
        ...,
        ext_modules=ext_modules
    )

If you want to do an automatic search for the highest supported C++ standard,
that is supported via a ``build_ext`` command override; it will only affect
``Pybind11Extensions``:

.. code-block:: python

    from setuptools import setup
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    ext_modules = [
        Pybind11Extension(
            "python_example",
            ["src/main.cpp"],
        ),
    ]

    setup(
        ...,
        cmdclass={"build_ext": build_ext},
        ext_modules=ext_modules
    )

.. _setup_helpers-pep518:

PEP 518 requirements (Pip 10+ required)
---------------------------------------

If you use `PEP 518's <https://www.python.org/dev/peps/pep-0518/>`_
``pyproject.toml`` file, you can ensure that ``pybind11`` is available during
the compilation of your project.  When this file exists, Pip will make a new
virtual environment, download just the packages listed here in ``requires=``,
and build a wheel (binary Python package). It will then throw away the
environment, and install your wheel.

Your ``pyproject.toml`` file will likely look something like this:

.. code-block:: toml

    [build-system]
    requires = ["setuptools", "wheel", "pybind11==2.6.0"]
    build-backend = "setuptools.build_meta"

.. note::

    The main drawback to this method is that a `PEP 517`_ compliant build tool,
    such as Pip 10+, is required for this approach to work; older versions of
    Pip completely ignore this file. If you distribute binaries (called wheels
    in Python) using something like `cibuildwheel`_, remember that ``setup.py``
    and ``pyproject.toml`` are not even contained in the wheel, so this high
    Pip requirement is only for source builds, and will not affect users of
    your binary wheels.

.. _PEP 517: https://www.python.org/dev/peps/pep-0517/
.. _cibuildwheel: https://cibuildwheel.readthedocs.io

.. _setup_helpers-setup_requires:

Classic ``setup_requires``
--------------------------

If you want to support old versions of Pip with the classic
``setup_requires=["pybind11"]`` keyword argument to setup, which triggers a
two-phase ``setup.py`` run, then you will need to use something like this to
ensure the first pass works (which has not yet installed the ``setup_requires``
packages, since it can't install something it does not know about):

.. code-block:: python

    try:
        from pybind11.setup_helpers import Pybind11Extension
    except ImportError:
        from setuptools import Extension as Pybind11Extension


It doesn't matter that the Extension class is not the enhanced subclass for the
first pass run; and the second pass will have the ``setup_requires``
requirements.

This is obviously more of a hack than the PEP 518 method, but it supports
ancient versions of Pip.

.. _setup_helpers-copy-manually:

Copy manually
-------------

You can also copy ``setup_helpers.py`` directly to your project; it was
designed to be usable standalone, like the old example ``setup.py``. You can
set ``include_pybind11=False`` to skip including the pybind11 package headers,
so you can use it with git submodules and a specific git version. If you use
this, you will need to import from a local file in ``setup.py`` and ensure the
helper file is part of your MANIFEST.


.. versionchanged:: 2.6

    Added ``setup_helpers`` file.

Building with cppimport
========================

[cppimport]_ is a small Python import hook that determines whether there is a C++
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

    cmake_minimum_required(VERSION 3.4...3.18)
    project(example LANGUAGES CXX)

    add_subdirectory(pybind11)
    pybind11_add_module(example example.cpp)

This assumes that the pybind11 repository is located in a subdirectory named
:file:`pybind11` and that the code is located in a file named :file:`example.cpp`.
The CMake command ``add_subdirectory`` will import the pybind11 project which
provides the ``pybind11_add_module`` function. It will take care of all the
details needed to build a Python extension module on any platform.

A working sample project, including a way to invoke CMake from :file:`setup.py` for
PyPI integration, can be found in the [cmake_example]_  repository.

.. [cmake_example] https://github.com/pybind/cmake_example

.. versionchanged:: 2.6
   CMake 3.4+ is required.

pybind11_add_module
-------------------

To ease the creation of Python extension modules, pybind11 provides a CMake
function with the following signature:

.. code-block:: cmake

    pybind11_add_module(<name> [MODULE | SHARED] [EXCLUDE_FROM_ALL]
                        [NO_EXTRAS] [THIN_LTO] [OPT_SIZE] source1 [source2 ...])

This function behaves very much like CMake's builtin ``add_library`` (in fact,
it's a wrapper function around that command). It will add a library target
called ``<name>`` to be built from the listed source files. In addition, it
will take care of all the Python-specific compiler and linker flags as well
as the OS- and Python-version-specific file extension. The produced target
``<name>`` can be further manipulated with regular CMake commands.

``MODULE`` or ``SHARED`` may be given to specify the type of library. If no
type is given, ``MODULE`` is used by default which ensures the creation of a
Python-exclusive module. Specifying ``SHARED`` will create a more traditional
dynamic library which can also be linked from elsewhere. ``EXCLUDE_FROM_ALL``
removes this target from the default build (see CMake docs for details).

Since pybind11 is a template library, ``pybind11_add_module`` adds compiler
flags to ensure high quality code generation without bloat arising from long
symbol names and duplication of code in different translation units. It
sets default visibility to *hidden*, which is required for some pybind11
features and functionality when attempting to load multiple pybind11 modules
compiled under different pybind11 versions.  It also adds additional flags
enabling LTO (Link Time Optimization) and strip unneeded symbols. See the
:ref:`FAQ entry <faq:symhidden>` for a more detailed explanation. These
latter optimizations are never applied in ``Debug`` mode.  If ``NO_EXTRAS`` is
given, they will always be disabled, even in ``Release`` mode. However, this
will result in code bloat and is generally not recommended.

As stated above, LTO is enabled by default. Some newer compilers also support
different flavors of LTO such as `ThinLTO`_. Setting ``THIN_LTO`` will cause
the function to prefer this flavor if available. The function falls back to
regular LTO if ``-flto=thin`` is not available. If
``CMAKE_INTERPROCEDURAL_OPTIMIZATION`` is set (either ON or OFF), then that
will be respected instead of the built-in flag search.

The ``OPT_SIZE`` flag enables size-based optimization equivalent to the
standard ``/Os`` or ``-Os`` compiler flags and the ``MinSizeRel`` build type,
which avoid optimizations that that can substantially increase the size of the
resulting binary. This flag is particularly useful in projects that are split
into performance-critical parts and associated bindings. In this case, we can
compile the project in release mode (and hence, optimize performance globally),
and specify ``OPT_SIZE`` for the binding target, where size might be the main
concern as performance is often less critical here. A ~25% size reduction has
been observed in practice. This flag only changes the optimization behavior at
a per-target level and takes precedence over the global CMake build type
(``Release``, ``RelWithDebInfo``) except for ``Debug`` builds, where
optimizations remain disabled.

.. _ThinLTO: http://clang.llvm.org/docs/ThinLTO.html

Configuration variables
-----------------------

By default, pybind11 will compile modules with the compiler default or the
minimum standard required by pybind11, whichever is higher.  You can set the
standard explicitly with
`CMAKE_CXX_STANDARD <https://cmake.org/cmake/help/latest/variable/CMAKE_CXX_STANDARD.html>`_:

.. code-block:: cmake

    set(CMAKE_CXX_STANDARD 14)  # or 11, 14, 17, 20
    set(CMAKE_CXX_STANDARD_REQUIRED ON)  # optional, ensure standard is supported
    set(CMAKE_CXX_EXTENSIONS OFF)  # optional, keep compiler extensionsn off


The variables can also be set when calling CMake from the command line using
the ``-D<variable>=<value>`` flag. You can also manually set ``CXX_STANDARD``
on a target or use ``target_compile_features`` on your targets - anything that
CMake supports.

Classic Python support: The target Python version can be selected by setting
``PYBIND11_PYTHON_VERSION`` or an exact Python installation can be specified
with ``PYTHON_EXECUTABLE``.  For example:

.. code-block:: bash

    cmake -DPYBIND11_PYTHON_VERSION=3.6 ..

    # Another method:
    cmake -DPYTHON_EXECUTABLE=/path/to/python ..

    # This often is a good way to get the current Python, works in environments:
    cmake -DPYTHON_EXECUTABLE=$(python3 -c "import sys; print(sys.executable)") ..


find_package vs. add_subdirectory
---------------------------------

For CMake-based projects that don't include the pybind11 repository internally,
an external installation can be detected through ``find_package(pybind11)``.
See the `Config file`_ docstring for details of relevant CMake variables.

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.4...3.18)
    project(example LANGUAGES CXX)

    find_package(pybind11 REQUIRED)
    pybind11_add_module(example example.cpp)

Note that ``find_package(pybind11)`` will only work correctly if pybind11
has been correctly installed on the system, e. g. after downloading or cloning
the pybind11 repository  :

.. code-block:: bash

    # Classic CMake
    cd pybind11
    mkdir build
    cd build
    cmake ..
    make install

    # CMake 3.15+
    cd pybind11
    cmake -S . -B build
    cmake --build build -j 2  # Build on 2 cores
    cmake --install build

Once detected, the aforementioned ``pybind11_add_module`` can be employed as
before. The function usage and configuration variables are identical no matter
if pybind11 is added as a subdirectory or found as an installed package. You
can refer to the same [cmake_example]_ repository for a full sample project
-- just swap out ``add_subdirectory`` for ``find_package``.

.. _Config file: https://github.com/pybind/pybind11/blob/master/tools/pybind11Config.cmake.in


.. _find-python-mode:

FindPython mode
---------------

CMake 3.12+ (3.15+ recommended) added a new module called FindPython that had a
highly improved search algorithm and modern targets and tools. If you use
FindPython, pybind11 will detect this and use the existing targets instead:

.. code-block:: cmake

    cmake_minumum_required(VERSION 3.15...3.18)
    project(example LANGUAGES CXX)

    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    # or add_subdirectory(pybind11)

    pybind11_add_module(example example.cpp)

You can also use the targets (as listed below) with FindPython. If you define
``PYBIND11_FINDPYTHON``, pybind11 will perform the FindPython step for you
(mostly useful when building pybind11's own tests, or as a way to change search
algorithms from the CMake invocation, with ``-DPYBIND11_FINDPYTHON=ON``.

.. warning::

    If you use FindPython2 and FindPython3 to dual-target Python, use the
    individual targets listed below, and avoid targets that directly include
    Python parts.

There are `many ways to hint or force a discovery of a specific Python
installation <https://cmake.org/cmake/help/latest/module/FindPython.html>`_),
setting ``Python_ROOT_DIR`` may be the most common one (though with
virtualenv/venv support, and Conda support, this tends to find the correct
Python version more often than the old system did).

.. versionadded:: 2.6

Advanced: interface library targets
-----------------------------------

Pybind11 supports modern CMake usage patterns with a set of interface targets,
available in all modes. The targets provided are:

   ``pybind11::headers``
     Just the pybind11 headers and minimum compile requirements

   ``pybind11::python2_no_register``
     Quiets the warning/error when mixing C++14 or higher and Python 2

   ``pybind11::pybind11``
     Python headers + ``pybind11::headers`` + ``pybind11::python2_no_register`` (Python 2 only)

   ``pybind11::python_link_helper``
     Just the "linking" part of pybind11:module

   ``pybind11::module``
     Everything for extension modules - ``pybind11::pybind11`` + ``Python::Module`` (FindPython CMake 3.15+) or ``pybind11::python_link_helper``

   ``pybind11::embed``
     Everything for embedding the Python interpreter - ``pybind11::pybind11`` + ``Python::Embed`` (FindPython) or Python libs

   ``pybind11::lto`` / ``pybind11::thin_lto``
     An alternative to `INTERPROCEDURAL_OPTIMIZATION` for adding link-time optimization.

   ``pybind11::windows_extras``
     ``/bigobj`` and ``/mp`` for MSVC.

   ``pybind11::opt_size``
     ``/Os`` for MSVC, ``-Os`` for other compilers. Does nothing for debug builds.

Two helper functions are also provided:

    ``pybind11_strip(target)``
      Strips a target (uses ``CMAKE_STRIP`` after the target is built)

    ``pybind11_extension(target)``
      Sets the correct extension (with SOABI) for a target.

You can use these targets to build complex applications. For example, the
``add_python_module`` function is identical to:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.4)
    project(example LANGUAGES CXX)

    find_package(pybind11 REQUIRED)  # or add_subdirectory(pybind11)

    add_library(example MODULE main.cpp)

    target_link_libraries(example PRIVATE pybind11::module pybind11::lto pybind11::windows_extras)

    pybind11_extension(example)
    pybind11_strip(example)

    set_target_properties(example PROPERTIES CXX_VISIBILITY_PRESET "hidden"
                                             CUDA_VISIBILITY_PRESET "hidden")

Instead of setting properties, you can set ``CMAKE_*`` variables to initialize these correctly.

.. warning::

    Since pybind11 is a metatemplate library, it is crucial that certain
    compiler flags are provided to ensure high quality code generation. In
    contrast to the ``pybind11_add_module()`` command, the CMake interface
    provides a *composable* set of targets to ensure that you retain flexibility.
    It can be expecially important to provide or set these properties; the
    :ref:`FAQ <faq:symhidden>` contains an explanation on why these are needed.

.. versionadded:: 2.6

.. _nopython-mode:

Advanced: NOPYTHON mode
-----------------------

If you want complete control, you can set ``PYBIND11_NOPYTHON`` to completely
disable Python integration (this also happens if you run ``FindPython2`` and
``FindPython3`` without running ``FindPython``). This gives you complete
freedom to integrate into an existing system (like `Scikit-Build's
<https://scikit-build.readthedocs.io>`_ ``PythonExtensions``).
``pybind11_add_module`` and ``pybind11_extension`` will be unavailable, and the
targets will be missing any Python specific behavior.

.. versionadded:: 2.6

Embedding the Python interpreter
--------------------------------

In addition to extension modules, pybind11 also supports embedding Python into
a C++ executable or library. In CMake, simply link with the ``pybind11::embed``
target. It provides everything needed to get the interpreter running. The Python
headers and libraries are attached to the target. Unlike ``pybind11::module``,
there is no need to manually set any additional properties here. For more
information about usage in C++, see :doc:`/advanced/embedding`.

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.4...3.18)
    project(example LANGUAGES CXX)

    find_package(pybind11 REQUIRED)  # or add_subdirectory(pybind11)

    add_executable(example main.cpp)
    target_link_libraries(example PRIVATE pybind11::embed)

.. _building_manually:

Building manually
=================

pybind11 is a header-only library, hence it is not necessary to link against
any special libraries and there are no intermediate (magic) translation steps.

On Linux, you can compile an example such as the one given in
:ref:`simple_example` using the following command:

.. code-block:: bash

    $ c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` example.cpp -o example`python3-config --extension-suffix`

The flags given here assume that you're using Python 3. For Python 2, just
change the executable appropriately (to ``python`` or ``python2``).

The ``python3 -m pybind11 --includes`` command fetches the include paths for
both pybind11 and Python headers. This assumes that pybind11 has been installed
using ``pip`` or ``conda``. If it hasn't, you can also manually specify
``-I <path-to-pybind11>/include`` together with the Python includes path
``python3-config --includes``.

Note that Python 2.7 modules don't use a special suffix, so you should simply
use ``example.so`` instead of ``example`python3-config --extension-suffix```.
Besides, the ``--extension-suffix`` option may or may not be available, depending
on the distribution; in the latter case, the module extension can be manually
set to ``.so``.

On macOS: the build command is almost the same but it also requires passing
the ``-undefined dynamic_lookup`` flag so as to ignore missing symbols when
building the module:

.. code-block:: bash

    $ c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` example.cpp -o example`python3-config --extension-suffix`

In general, it is advisable to include several additional build parameters
that can considerably reduce the size of the created binary. Refer to section
:ref:`cmake` for a detailed example of a suitable cross-platform CMake-based
build system that works on all platforms including Windows.

.. note::

    On Linux and macOS, it's better to (intentionally) not link against
    ``libpython``. The symbols will be resolved when the extension library
    is loaded into a Python binary. This is preferable because you might
    have several different installations of a given Python version (e.g. the
    system-provided Python, and one that ships with a piece of commercial
    software). In this way, the plugin will work with both versions, instead
    of possibly importing a second Python library into a process that already
    contains one (which will lead to a segfault).


Building with vcpkg
===================
You can download and install pybind11 using the Microsoft `vcpkg
<https://github.com/Microsoft/vcpkg/>`_ dependency manager:

.. code-block:: bash

    git clone https://github.com/Microsoft/vcpkg.git
    cd vcpkg
    ./bootstrap-vcpkg.sh
    ./vcpkg integrate install
    vcpkg install pybind11

The pybind11 port in vcpkg is kept up to date by Microsoft team members and
community contributors. If the version is out of date, please `create an issue
or pull request <https://github.com/Microsoft/vcpkg/>`_ on the vcpkg
repository.

Generating binding code automatically
=====================================

The ``Binder`` project is a tool for automatic generation of pybind11 binding
code by introspecting existing C++ codebases using LLVM/Clang. See the
[binder]_ documentation for details.

.. [binder] http://cppbinder.readthedocs.io/en/latest/about.html

[AutoWIG]_ is a Python library that wraps automatically compiled libraries into
high-level languages. It parses C++ code using LLVM/Clang technologies and
generates the wrappers using the Mako templating engine. The approach is automatic,
extensible, and applies to very complex C++ libraries, composed of thousands of
classes or incorporating modern meta-programming constructs.

.. [AutoWIG] https://github.com/StatisKit/AutoWIG

[robotpy-build]_ is a is a pure python, cross platform build tool that aims to
simplify creation of python wheels for pybind11 projects, and provide
cross-project dependency management. Additionally, it is able to autogenerate
customizable pybind11-based wrappers by parsing C++ header files.

.. [robotpy-build] https://robotpy-build.readthedocs.io
