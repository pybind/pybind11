.. _deprecated:

Deprecated
##########

Support for C++11 is deprecated and will be removed in a future version. Please
use at least C++14.

Support for FindPythonLibs (not available in CMake 3.26+ mode) is deprecated
and will be removed in a future version. The default mode is also going to
change to ``"new"`` from ``"compat"`` in the future.

The following features were deprecated before pybind11 3.0, and may be removed
in minor releases of pybind11 3.x. "Compiler" means a ``[[deprecated]]``
warning at compile time. Items that started to warn in 3.x will be kept
until at least pybind11 4.0.

.. list-table:: Deprecated before 3.0
   :header-rows: 1
   :widths: 30 30 8 8 14

   * - Feature
     - Replacement
     - Version
     - Year
     - Warning
   * - ``handle::call(...)``
     - ``operator()(...)``
     - 1.8
     - 2016
     - Compiler
   * - ``py::base<T>()``
     - ``py::class_<T, Base>``
     - 2.0
     - 2016
     - Compiler
   * - ``handle::check()``
     - ``handle::operator bool()``
     - 2.0
     - 2016
     - Compiler
   * - ``Name::check()`` on ``py::object`` subclasses
     - ``py::isinstance<Name>(obj)``
     - 2.0
     - 2016
     - Compiler
   * - ``object(handle, bool)`` and ``Name(handle, bool)``
     - ``py::reinterpret_borrow`` / ``py::reinterpret_steal``
     - 2.0
     - 2016
     - Compiler
   * - ``capsule(PyObject *, bool)``
     - ``py::reinterpret_borrow`` / ``py::reinterpret_steal``
     - 2.0
     - 2016
     - Compiler
   * - ``array_t<T>(handle, bool)``
     - ``array_t<T>::ensure()``
     - 2.0
     - 2016
     - Compiler
   * - ``obj.attr(...)`` as ``bool``
     - ``py::hasattr(obj, ...)``
     - 2.0
     - 2016
     - Compiler
   * - ``obj[key]`` as ``bool``
     - ``obj.contains(key)``
     - 2.0
     - 2016
     - Compiler
   * - ``obj.str()``
     - ``py::str(obj)``
     - 2.1
     - 2017
     - Compiler
   * - ``py::metaclass()`` without arguments
     - Remove it (default since 2.1)
     - 2.1
     - 2017
     - Compiler
   * - ``handle::operator==`` / ``operator!=``
     - ``obj1.is(obj2)``
     - 2.2
     - 2017
     - Compiler
   * - ``error_already_set::clear()``
     - Remove it (does nothing)
     - 2.2
     - 2017
     - Compiler
   * - ``PYBIND11_PLUGIN``
     - ``PYBIND11_MODULE``
     - 2.2
     - 2017
     - Compiler
   * - ``#include <pybind11/common.h>``
     - ``#include <pybind11/pybind11.h>``
     - 2.2.1
     - 2017
     - ``#warning``
   * - Old-style placement-new ``__init__`` / ``__setstate__``
     - ``py::init(...)`` / ``py::pickle(...)``
     - 2.2
     - 2017
     - Runtime, debug builds only
   * - ``py::module_(name, doc)`` constructor
     - ``PYBIND11_MODULE`` or ``module_::create_extension_module``
     - 2.6
     - 2020
     - Compiler
   * - ``get_type_overload``
     - ``get_override``
     - 2.6
     - 2020
     - Compiler
   * - ``obj.get_type()``
     - ``py::type::handle_of(obj)``
     - 2.6
     - 2020
     - Compiler (since 3.0)
   * - ``PYBIND11_OVERLOAD*`` macros, ``get_overload``
     - ``PYBIND11_OVERRIDE*``, ``get_override``
     - 2.6
     - 2020
     - Compiler (since 3.2)
   * - ``PYBIND11_CPP_STANDARD`` (CMake)
     - ``CMAKE_CXX_STANDARD``
     - 2.6
     - 2020
     - CMake
   * - ``pybind11_add_module(... SYSTEM)`` (CMake)
     - Remove it (does nothing)
     - 2.6
     - 2020
     - CMake
   * - ``PYTHON_VERSION`` as an input (CMake)
     - ``PYBIND11_PYTHON_VERSION``
     - 2.6
     - 2020
     - CMake
   * - ``py::capsule(value, destructor)``
     - ``py::capsule(value, name, destructor)``
     - 2.10.1
     - 2022
     - Compiler
   * - ``py::exception<T>::operator()(message)``
     - ``py::set_error(exc, message)``
     - 2.12
     - 2024
     - Compiler
   * - ``py::arg_t<T>``
     - ``py::arg_v``
     - 2.0
     - 2016
     - Compiler (since 3.2)
   * - ``py::detail::_(...)``
     - ``py::detail::const_name(...)``
     - 2.9.1
     - 2022
     - Compiler (since 3.2)

The following features were deprecated in pybind11 3.0 or later. They will be
kept until at least pybind11 4.0.

.. list-table:: Deprecated in 3.x
   :header-rows: 1
   :widths: 30 30 8 8 14

   * - Feature
     - Replacement
     - Version
     - Year
     - Warning
   * - FindPythonLibs mode (CMake)
     - FindPython mode (``PYBIND11_FINDPYTHON=ON``)
     - 3.0
     - 2025
     - None
   * - ``py::enum_``
     - ``py::native_enum``
     - 3.0
     - 2025
     - None
   * - ``py::module_::module_def``
     - ``PyModuleDef``
     - 3.0
     - 2025
     - Compiler (since 3.2)
   * - ``py::mod_gil_not_used(bool)``
     - ``py::mod_gil_not_used()`` / ``py::mod_gil_used()``
     - 3.1
     - 2026
     - Compiler


.. _deprecated_enum:

``py::enum_``
=============

This is the original documentation for ``py::enum_``, which is deprecated
because it is not `PEP 435 compatible <https://peps.python.org/pep-0435/>`_
(see also `#2332 <https://github.com/pybind/pybind11/issues/2332>`_).
Please prefer ``py::native_enum`` (added with pybind11v3) when writing
new bindings. See :ref:`native_enum` for more information.

Let's suppose that we have an example class that contains internal types
like enumerations, e.g.:

.. code-block:: cpp

    struct Pet {
        enum Kind {
            Dog = 0,
            Cat
        };

        struct Attributes {
            float age = 0;
        };

        Pet(const std::string &name, Kind type) : name(name), type(type) { }

        std::string name;
        Kind type;
        Attributes attr;
    };

The binding code for this example looks as follows:

.. code-block:: cpp

    py::class_<Pet> pet(m, "Pet");

    pet.def(py::init<const std::string &, Pet::Kind>())
        .def_readwrite("name", &Pet::name)
        .def_readwrite("type", &Pet::type)
        .def_readwrite("attr", &Pet::attr);

    py::enum_<Pet::Kind>(pet, "Kind")
        .value("Dog", Pet::Kind::Dog)
        .value("Cat", Pet::Kind::Cat)
        .export_values();

    py::class_<Pet::Attributes>(pet, "Attributes")
        .def(py::init<>())
        .def_readwrite("age", &Pet::Attributes::age);


To ensure that the nested types ``Kind`` and ``Attributes`` are created within the scope of ``Pet``, the
``pet`` ``py::class_`` instance must be supplied to the :class:`enum_` and ``py::class_``
constructor. The :func:`enum_::export_values` function exports the enum entries
into the parent scope, which should be skipped for newer C++11-style strongly
typed enums.

.. code-block:: pycon

    >>> p = Pet("Lucy", Pet.Cat)
    >>> p.type
    Kind.Cat
    >>> int(p.type)
    1L

The entries defined by the enumeration type are exposed in the ``__members__`` property:

.. code-block:: pycon

    >>> Pet.Kind.__members__
    {'Dog': Kind.Dog, 'Cat': Kind.Cat}

The ``name`` property returns the name of the enum value as a unicode string.

.. note::

    It is also possible to use ``str(enum)``, however these accomplish different
    goals. The following shows how these two approaches differ.

    .. code-block:: pycon

        >>> p = Pet("Lucy", Pet.Cat)
        >>> pet_type = p.type
        >>> pet_type
        Pet.Cat
        >>> str(pet_type)
        'Pet.Cat'
        >>> pet_type.name
        'Cat'

.. note::

    When the special tag ``py::arithmetic()`` is specified to the ``enum_``
    constructor, pybind11 creates an enumeration that also supports rudimentary
    arithmetic and bit-level operations like comparisons, and, or, xor, negation,
    etc.

    .. code-block:: cpp

        py::enum_<Pet::Kind>(pet, "Kind", py::arithmetic())
           ...

    By default, these are omitted to conserve space.

.. warning::

    Contrary to Python customs, enum values from the wrappers should not be compared using ``is``, but with ``==`` (see `#1177 <https://github.com/pybind/pybind11/issues/1177>`_ for background).
