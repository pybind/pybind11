.. _changelog:

Changelog
#########

1.8 (April 30, 2016)
----------------------
TBD

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
