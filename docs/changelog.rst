.. _changelog:

Changelog
#########

Starting with version 1.8, pybind11 releases use a
[semantic versioning](http://semver.org) policy.

Breaking changes queued for v2.0.0 (Not yet released)
-----------------------------------------------------
* Redesigned virtual call mechanism and user-facing syntax (see
  https://github.com/pybind/pybind11/commit/86d825f3302701d81414ddd3d38bcd09433076bc)

* Remove ``handle.call()`` method

1.9.0 (Not yet released)
------------------------
* Queued changes: map indexing suite, documentation for indexing suites.
* Mapping a stateless C++ function to Python and back is now "for free" (i.e. no call overheads)
* Support for translation of arbitrary C++ exceptions to Python counterparts
* Added ``eval`` and ``eval_file`` functions for evaluating expressions and
  statements from a string or file
* eigen.h type converter fixed for non-contiguous arrays (e.g. slices)
* Print more informative error messages when ``make_tuple()`` or ``cast()`` fail
* ``std::enable_shared_from_this<>`` now also works for ``const`` values
* A return value policy can now be passed to ``handle::operator()``
* ``make_iterator()`` improvements for better compatibility with various types
  (now uses prefix increment operator); it now also accepts iterators with
  different begin/end types as long as they are equality comparable.
* ``arg()`` now accepts a wider range of argument types for default values
* Added ``repr()`` method to the ``handle`` class.
* Added support for registering structured dtypes via ``PYBIND11_NUMPY_DTYPE()`` macro.
* Added ``PYBIND11_STR_TYPE`` macro which maps to the ``builtins.str`` type.
* Added a simplified ``buffer_info`` constructor for 1-dimensional buffers.
* Format descriptor strings should now be accessed via ``format_descriptor::format()``
  (for compatibility purposes, the old syntax ``format_descriptor::value`` will still
  work for non-structured data types).
* Added a class wrapping NumPy array descriptors: ``dtype``.
* Added buffer/NumPy support for ``char[N]`` and ``std::array<char, N>`` types.
* ``array`` gained new constructors accepting dtype objects.
* Added constructors for ``array`` and ``array_t`` explicitly accepting shape and
  strides; if strides are not provided, they are deduced assuming C-contiguity.
  Also added simplified constructors for 1-dimensional case.
* Added constructors for ``str`` from ``bytes`` and for ``bytes`` from ``str``.
  This will do the UTF-8 decoding/encoding as required.
* Added constructors for ``str`` and ``bytes`` from zero-terminated char pointers,
  and from char pointers and length.
* Added ``memoryview`` wrapper type which is constructible from ``buffer_info``.
* Various minor improvements of library internals (no user-visible changes)

1.8.1 (July 12, 2016)
----------------------
* Fixed a rare but potentially very severe issue when the garbage collector ran
  during pybind11 type creation.

1.8.0 (June 14, 2016)
----------------------
* Redesigned CMake build system which exports a convenient
  ``pybind11_add_module`` function to parent projects.
* ``std::vector<>`` type bindings analogous to Boost.Python's ``indexing_suite``
* Transparent conversion of sparse and dense Eigen matrices and vectors (``eigen.h``)
* Added an ``ExtraFlags`` template argument to the NumPy ``array_t<>`` wrapper
  to disable an enforced cast that may lose precision, e.g. to create overloads
  for different precisions and complex vs real-valued matrices.
* Prevent implicit conversion of floating point values to integral types in
  function arguments
* Fixed incorrect default return value policy for functions returning a shared
  pointer
* Don't allow registering a type via ``class_`` twice
* Don't allow casting a ``None`` value into a C++ lvalue reference
* Fixed a crash in ``enum_::operator==`` that was triggered by the ``help()`` command
* Improved detection of whether or not custom C++ types can be copy/move-constructed
* Extended ``str`` type to also work with ``bytes`` instances
* Added a ``"name"_a`` user defined string literal that is equivalent to ``py::arg("name")``.
* When specifying function arguments via ``py::arg``, the test that verifies
  the number of arguments now runs at compile time.
* Added ``[[noreturn]]`` attribute to ``pybind11_fail()`` to quench some
  compiler warnings
* List function arguments in exception text when the dispatch code cannot find
  a matching overload
* Added ``PYBIND11_OVERLOAD_NAME`` and ``PYBIND11_OVERLOAD_PURE_NAME`` macros which
  can be used to override virtual methods whose name differs in C++ and Python
  (e.g. ``__call__`` and ``operator()``)
* Various minor ``iterator`` and ``make_iterator()`` improvements
* Transparently support ``__bool__`` on Python 2.x and Python 3.x
* Fixed issue with destructor of unpickled object not being called
* Minor CMake build system improvements on Windows
* New ``pybind11::args`` and ``pybind11::kwargs`` types to create functions which
  take an arbitrary number of arguments and keyword arguments
* New syntax to call a Python function from C++ using ``*args`` and ``*kwargs``
* The functions ``def_property_*`` now correctly process docstring arguments (these
  formerly caused a segmentation fault)
* Many ``mkdoc.py`` improvements (enumerations, template arguments, ``DOC()``
  macro accepts more arguments)
* Cygwin support
* Documentation improvements (pickling support, ``keep_alive``, macro usage)

1.7 (April 30, 2016)
----------------------
* Added a new ``move`` return value policy that triggers C++11 move semantics.
  The automatic return value policy falls back to this case whenever a rvalue
  reference is encountered
* Significantly more general GIL state routines that are used instead of
  Python's troublesome ``PyGILState_Ensure`` and ``PyGILState_Release`` API
* Redesign of opaque types that drastically simplifies their usage
* Extended ability to pass values of type ``[const] void *``
* ``keep_alive`` fix: don't fail when there is no patient
* ``functional.h``: acquire the GIL before calling a Python function
* Added Python RAII type wrappers ``none`` and ``iterable``
* Added ``*args`` and ``*kwargs`` pass-through parameters to
  ``pybind11.get_include()`` function
* Iterator improvements and fixes
* Documentation on return value policies and opaque types improved

1.6 (April 30, 2016)
----------------------
* Skipped due to upload to PyPI gone wrong and inability to recover
  (https://github.com/pypa/packaging-problems/issues/74)

1.5 (April 21, 2016)
----------------------
* For polymorphic types, use RTTI to try to return the closest type registered with pybind11
* Pickling support for serializing and unserializing C++ instances to a byte stream in Python
* Added a convenience routine ``make_iterator()`` which turns a range indicated
  by a pair of C++ iterators into a iterable Python object
* Added ``len()`` and a variadic ``make_tuple()`` function
* Addressed a rare issue that could confuse the current virtual function
  dispatcher and another that could lead to crashes in multi-threaded
  applications
* Added a ``get_include()`` function to the Python module that returns the path
  of the directory containing the installed pybind11 header files
* Documentation improvements: import issues, symbol visibility, pickling, limitations
* Added casting support for ``std::reference_wrapper<>``

1.4 (April 7, 2016)
--------------------------
* Transparent type conversion for ``std::wstring`` and ``wchar_t``
* Allow passing ``nullptr``-valued strings
* Transparent passing of ``void *`` pointers using capsules
* Transparent support for returning values wrapped in ``std::unique_ptr<>``
* Improved docstring generation for compatibility with Sphinx
* Nicer debug error message when default parameter construction fails
* Support for "opaque" types that bypass the transparent conversion layer for STL containers
* Redesigned type casting interface to avoid ambiguities that could occasionally cause compiler errors
* Redesigned property implementation; fixes crashes due to an unfortunate default return value policy
* Anaconda package generation support

1.3 (March 8, 2016)
--------------------------

* Added support for the Intel C++ compiler (v15+)
* Added support for the STL unordered set/map data structures
* Added support for the STL linked list data structure
* NumPy-style broadcasting support in ``pybind11::vectorize``
* pybind11 now displays more verbose error messages when ``arg::operator=()`` fails
* pybind11 internal data structures now live in a version-dependent namespace to avoid ABI issues
* Many, many bugfixes involving corner cases and advanced usage

1.2 (February 7, 2016)
--------------------------

* Optional: efficient generation of function signatures at compile time using C++14
* Switched to a simpler and more general way of dealing with function default
  arguments. Unused keyword arguments in function calls are now detected and
  cause errors as expected
* New ``keep_alive`` call policy analogous to Boost.Python's ``with_custodian_and_ward``
* New ``pybind11::base<>`` attribute to indicate a subclass relationship
* Improved interface for RAII type wrappers in ``pytypes.h``
* Use RAII type wrappers consistently within pybind11 itself. This
  fixes various potential refcount leaks when exceptions occur
* Added new ``bytes`` RAII type wrapper (maps to ``string`` in Python 2.7)
* Made handle and related RAII classes const correct, using them more
  consistently everywhere now
* Got rid of the ugly ``__pybind11__`` attributes on the Python side---they are
  now stored in a C++ hash table that is not visible in Python
* Fixed refcount leaks involving NumPy arrays and bound functions
* Vastly improved handling of shared/smart pointers
* Removed an unnecessary copy operation in ``pybind11::vectorize``
* Fixed naming clashes when both pybind11 and NumPy headers are included
* Added conversions for additional exception types
* Documentation improvements (using multiple extension modules, smart pointers,
  other minor clarifications)
* unified infrastructure for parsing variadic arguments in ``class_`` and cpp_function
* Fixed license text (was: ZLIB, should have been: 3-clause BSD)
* Python 3.2 compatibility
* Fixed remaining issues when accessing types in another plugin module
* Added enum comparison and casting methods
* Improved SFINAE-based detection of whether types are copy-constructible
* Eliminated many warnings about unused variables and the use of ``offsetof()``
* Support for ``std::array<>`` conversions

1.1 (December 7, 2015)
--------------------------

* Documentation improvements (GIL, wrapping functions, casting, fixed many typos)
* Generalized conversion of integer types
* Improved support for casting function objects
* Improved support for ``std::shared_ptr<>`` conversions
* Initial support for ``std::set<>`` conversions
* Fixed type resolution issue for types defined in a separate plugin module
* Cmake build system improvements
* Factored out generic functionality to non-templated code (smaller code size)
* Added a code size / compile time benchmark vs Boost.Python
* Added an appveyor CI script

1.0 (October 15, 2015)
------------------------
* Initial release
