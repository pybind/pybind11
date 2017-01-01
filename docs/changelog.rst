.. _changelog:

Changelog
#########

Starting with version 1.8.0, pybind11 releases use a `semantic versioning
<http://semver.org>`_ policy.

v2.0.0 (Jan 1, 2017)
-----------------------------------------------------

* Fixed a reference counting regression affecting types with custom metaclasses
  (introduced in v2.0.0-rc1).
  `#571 <https://github.com/pybind/pybind11/pull/571>`_.

* Quenched a CMake policy warning.
  `#570 <https://github.com/pybind/pybind11/pull/570>`_.

v2.0.0-rc1 (Dec 23, 2016)
-----------------------------------------------------

The pybind11 developers are excited to issue a release candidate of pybind11
with a subsequent v2.0.0 release planned in early January next year.

An incredible amount of effort by went into pybind11 over the last ~5 months,
leading to a release that is jam-packed with exciting new features and numerous
usuability improvements. The following list links PRs or individual commits
whenever applicable.

Happy Christmas!

* Support for binding C++ class hierarchies that make use of multiple
  inheritance. `#410 <https://github.com/pybind/pybind11/pull/410>`_.

* PyPy support: pybind11 now supports nightly builds of PyPy and will
  interoperate with the future 5.7 release. No code changes are necessary,
  everything "just" works as usual. Note that we only target the Python 2.7
  branch for now; support for 3.x will be added once its ``cpyext`` extension
  support catches up. A few minor features remain unsupported for the time
  being (notably dynamic attributes in custom types).
  `#527 <https://github.com/pybind/pybind11/pull/527>`_.

* Significant work on the documentation -- in particular, the monolitic
  ``advanced.rst`` file was restructured into a easier to read hierarchical
  organization. `#448 <https://github.com/pybind/pybind11/pull/448>`_.

* Many NumPy-related improvements:

  1. Object-oriented API to access and modify NumPy ``ndarray`` instances,
     replicating much of the corresponding NumPy C API functionality.
     `#402 <https://github.com/pybind/pybind11/pull/402>`_.

  2. NumPy array ``dtype`` array descriptors are now first-class citizens and
     are exposed via a new class ``py::dtype``.

  3. Structured dtypes can be registered using the ``PYBIND11_NUMPY_DTYPE()``
     macro. Special ``array`` constructors accepting dtype objects were also
     added.

     One potential caveat involving this change: format descriptor strings
     should now be accessed via ``format_descriptor::format()`` (however, for
     compatibility purposes, the old syntax ``format_descriptor::value`` will
     still work for non-structured data types). `#308
     <https://github.com/pybind/pybind11/pull/308>`_.

  4. Further improvements to support structured dtypes throughout the system.
     `#472 <https://github.com/pybind/pybind11/pull/472>`_,
     `#474 <https://github.com/pybind/pybind11/pull/474>`_,
     `#459 <https://github.com/pybind/pybind11/pull/459>`_,
     `#453 <https://github.com/pybind/pybind11/pull/453>`_,
     `#452 <https://github.com/pybind/pybind11/pull/452>`_, and
     `#505 <https://github.com/pybind/pybind11/pull/505>`_.

  5. Fast access operators. `#497 <https://github.com/pybind/pybind11/pull/497>`_.

  6. Constructors for arrays whose storage is owned by another object.
     `#440 <https://github.com/pybind/pybind11/pull/440>`_.

  7. Added constructors for ``array`` and ``array_t`` explicitly accepting shape
     and strides; if strides are not provided, they are deduced assuming
     C-contiguity. Also added simplified constructors for 1-dimensional case.

  8. Added buffer/NumPy support for ``char[N]`` and ``std::array<char, N>`` types.

  9. Added ``memoryview`` wrapper type which is constructible from ``buffer_info``.

* Eigen: many additional conversions and support for non-contiguous
  arrays/slices.
  `#427 <https://github.com/pybind/pybind11/pull/427>`_,
  `#315 <https://github.com/pybind/pybind11/pull/315>`_,
  `#316 <https://github.com/pybind/pybind11/pull/316>`_,
  `#312 <https://github.com/pybind/pybind11/pull/312>`_, and
  `#267 <https://github.com/pybind/pybind11/pull/267>`_

* Incompatible changes in ``class_<...>::class_()``:

    1. Declarations of types that provide access via the buffer protocol must
       now include the ``py::buffer_protocol()`` annotation as an argument to
       the ``class_`` constructor.

    2. Declarations of types that require a custom metaclass (i.e. all classes
       which include static properties via commands such as
       ``def_readwrite_static()``) must now include the ``py::metaclass()``
       annotation as an argument to the ``class_`` constructor.

       These two changes were necessary to make type definitions in pybind11
       future-proof, and to support PyPy via its cpyext mechanism. `#527
       <https://github.com/pybind/pybind11/pull/527>`_.


    3. This version of pybind11 uses a redesigned mechnism for instantiating
       trempoline classes that are used to override virtual methods from within
       Python. This led to the following user-visible syntax change: instead of

       .. code-block:: cpp

           py::class_<TrampolineClass>("MyClass")
             .alias<MyClass>()
             ....

       write

       .. code-block:: cpp

           py::class_<MyClass, TrampolineClass>("MyClass")
             ....

       Importantly, both the original and the trampoline class are now
       specified as an arguments (in arbitrary order) to the ``py::class_``
       template, and the ``alias<..>()`` call is gone. The new scheme has zero
       overhead in cases when Python doesn't override any functions of the
       underlying C++ class. `rev. 86d825
       <https://github.com/pybind/pybind11/commit/86d825>`_.

* Added ``eval`` and ``eval_file`` functions for evaluating expressions and
  statements from a string or file. `rev. 0d3fc3
  <https://github.com/pybind/pybind11/commit/0d3fc3>`_.

* pybind11 can now create types with a modifiable dictionary.
  `#437 <https://github.com/pybind/pybind11/pull/437>`_ and
  `#444 <https://github.com/pybind/pybind11/pull/444>`_.

* Support for translation of arbitrary C++ exceptions to Python counterparts.
  `#296 <https://github.com/pybind/pybind11/pull/296>`_ and
  `#273 <https://github.com/pybind/pybind11/pull/273>`_.

* Report full backtraces through mixed C++/Python code, better reporting for
  import errors, fixed GIL management in exception processing.
  `#537 <https://github.com/pybind/pybind11/pull/537>`_,
  `#494 <https://github.com/pybind/pybind11/pull/494>`_,
  `rev. e72d95 <https://github.com/pybind/pybind11/commit/e72d95>`_, and
  `rev. 099d6e <https://github.com/pybind/pybind11/commit/099d6e>`_.

* Support for bit-level operations, comparisons, and serialization of C++
  enumerations. `#503 <https://github.com/pybind/pybind11/pull/503>`_,
  `#508 <https://github.com/pybind/pybind11/pull/508>`_,
  `#380 <https://github.com/pybind/pybind11/pull/380>`_,
  `#309 <https://github.com/pybind/pybind11/pull/309>`_.
  `#311 <https://github.com/pybind/pybind11/pull/311>`_.

* The ``class_`` constructor now accepts its template arguments in any order.
  `#385 <https://github.com/pybind/pybind11/pull/385>`_.

* Attribute and item accessors now have a more complete interface which makes
  it possible to chain attributes as in
  ``obj.attr("a")[key].attr("b").attr("method")(1, 2, 3)``. `#425
  <https://github.com/pybind/pybind11/pull/425>`_.

* Major redesign of the default and conversion constructors in ``pytypes.h``.
  `#464 <https://github.com/pybind/pybind11/pull/464>`_.

* Added built-in support for ``std::shared_ptr`` holder type. It is no longer
  necessary to to include a declaration of the form
  ``PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)`` (though continuing to
  do so won't cause an error).
  `#454 <https://github.com/pybind/pybind11/pull/454>`_.

* New ``py::overload_cast`` casting operator to select among multiple possible
  overloads of a function. An example:

    .. code-block:: cpp

        py::class_<Pet>(m, "Pet")
            .def("set", py::overload_cast<int>(&Pet::set), "Set the pet's age")
            .def("set", py::overload_cast<const std::string &>(&Pet::set), "Set the pet's name");

  This feature only works on C++14-capable compilers.
  `#541 <https://github.com/pybind/pybind11/pull/541>`_.

* C++ types are automatically cast to Python types, e.g. when assigning
  them as an attribute. For instance, the following is now legal:

    .. code-block:: cpp

        py::module m = /* ... */
        m.attr("constant") = 123;

  (Previously, a ``py::cast`` call was necessary to avoid a compilation error.)
  `#551 <https://github.com/pybind/pybind11/pull/551>`_.

* Redesigned ``pytest``-based test suite. `#321 <https://github.com/pybind/pybind11/pull/321>`_.

* Instance tracking to detect reference leaks in test suite. `#324 <https://github.com/pybind/pybind11/pull/324>`_

* pybind11 can now distinguish between multiple different instances that are
  located at the same memory address, but which have different types.
  `#329 <https://github.com/pybind/pybind11/pull/329>`_.

* Improved logic in ``move`` return value policy.
  `#510 <https://github.com/pybind/pybind11/pull/510>`_,
  `#297 <https://github.com/pybind/pybind11/pull/297>`_.

* Generalized unpacking API to permit calling Python functions from C++ using
  notation such as ``foo(a1, a2, *args, "ka"_a=1, "kb"_a=2, **kwargs)``. `#372 <https://github.com/pybind/pybind11/pull/372>`_.

* ``py::print()`` function whose behavior matches that of the native Python
  ``print()`` function. `#372 <https://github.com/pybind/pybind11/pull/372>`_.

* Added ``py::dict`` keyword constructor:``auto d = dict("number"_a=42,
  "name"_a="World");``. `#372 <https://github.com/pybind/pybind11/pull/372>`_.

* Added ``py::str::format()`` method and ``_s`` literal: ``py::str s = "1 + 2
  = {}"_s.format(3);``. `#372 <https://github.com/pybind/pybind11/pull/372>`_.

* Added ``py::repr()`` function which is equivalent to Python's builtin
  ``repr()``. `#333 <https://github.com/pybind/pybind11/pull/333>`_.

* Improved construction and destruction logic for holder types. It is now
  possible to reference instances with smart pointer holder types without
  constructing the holder if desired. The ``PYBIND11_DECLARE_HOLDER_TYPE``
  macro now accepts an optional second parameter to indicate whether the holder
  type uses intrusive reference counting.
  `#533 <https://github.com/pybind/pybind11/pull/533>`_ and
  `#561 <https://github.com/pybind/pybind11/pull/561>`_.

* Mapping a stateless C++ function to Python and back is now "for free" (i.e.
  no extra indirections or argument conversion overheads). `rev. 954b79
  <https://github.com/pybind/pybind11/commit/954b79>`_.

* Bindings for ``std::valarray<T>``.
  `#545 <https://github.com/pybind/pybind11/pull/545>`_.

* Improved support for C++17 capable compilers.
  `#562 <https://github.com/pybind/pybind11/pull/562>`_.

* Bindings for ``std::optional<t>``.
  `#475 <https://github.com/pybind/pybind11/pull/475>`_,
  `#476 <https://github.com/pybind/pybind11/pull/476>`_,
  `#479 <https://github.com/pybind/pybind11/pull/479>`_,
  `#499 <https://github.com/pybind/pybind11/pull/499>`_, and
  `#501 <https://github.com/pybind/pybind11/pull/501>`_.

* ``stl_bind.h``: general improvements and support for ``std::map`` and
  ``std::unordered_map``.
  `#490 <https://github.com/pybind/pybind11/pull/490>`_,
  `#282 <https://github.com/pybind/pybind11/pull/282>`_,
  `#235 <https://github.com/pybind/pybind11/pull/235>`_.

* The ``std::tuple``, ``std::pair``, ``std::list``, and ``std::vector`` type
  casters now accept any Python sequence type as input. `rev. 107285
  <https://github.com/pybind/pybind11/commit/107285>`_.

* Improved CMake Python detection on multi-architecture Linux.
  `#532 <https://github.com/pybind/pybind11/pull/532>`_.

* Infrastructure to selectively disable or enable parts of the automatically
  generated docstrings. `#486 <https://github.com/pybind/pybind11/pull/486>`_.

* ``reference`` and ``reference_internal`` are now the default return value
  properties for static and non-static properties, respectively. `#473
  <https://github.com/pybind/pybind11/pull/473>`_. (the previous defaults
  were ``automatic``). `#473 <https://github.com/pybind/pybind11/pull/473>`_.

* Support for ``std::unique_ptr`` with non-default deleters or no deleter at
  all (``py::nodelete``). `#384 <https://github.com/pybind/pybind11/pull/384>`_.

* Deprecated ``handle::call()`` method. The new syntax to call Python
  functions is simply ``handle()``. It can also be invoked explicitly via
  ``handle::operator<X>()``, where ``X`` is an optional return value policy.

* Print more informative error messages when ``make_tuple()`` or ``cast()``
  fail. `#262 <https://github.com/pybind/pybind11/pull/262>`_.

* Creation of holder types for classes deriving from
  ``std::enable_shared_from_this<>`` now also works for ``const`` values.
  `#260 <https://github.com/pybind/pybind11/pull/260>`_.

* ``make_iterator()`` improvements for better compatibility with various
  types (now uses prefix increment operator); it now also accepts iterators
  with different begin/end types as long as they are equality comparable.
  `#247 <https://github.com/pybind/pybind11/pull/247>`_.

* ``arg()`` now accepts a wider range of argument types for default values.
  `#244 <https://github.com/pybind/pybind11/pull/244>`_.

* Support ``keep_alive`` where the nurse object may be ``None``. `#341
  <https://github.com/pybind/pybind11/pull/341>`_.

* Added constructors for ``str`` and ``bytes`` from zero-terminated char
  pointers, and from char pointers and length. Added constructors for ``str``
  from ``bytes`` and for ``bytes`` from ``str``, which will perform UTF-8
  decoding/encoding as required.

* Many other improvements of library internals without user-visible changes


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
