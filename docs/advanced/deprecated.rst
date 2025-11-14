.. _deprecated:

Deprecated
##########

Support for Python 3.8 is deprecated and will be removed in 3.1.

Support for C++11 is deprecated and will be removed in a future version. Please
use at least C++14.

Support for FindPythonLibs (not available in CMake 3.26+ mode) is deprecated
and will be removed in a future version. The default mode is also going to
change to ``"new"`` from ``"compat"`` in the future.

The following features were deprecated before pybind11 3.0, and may be removed
in minor releases of pybind11 3.x.

.. list-table:: Deprecated Features
   :header-rows: 1
   :widths: 30 15 10

   * - Feature
     - Deprecated Version
     - Year
   * - ``py::metaclass()``
     - 2.1
     - 2017
   * - ``PYBIND11_PLUGIN``
     - 2.2
     - 2017
   * - ``py::set_error()`` replacing ``operator()``
     - 2.12
     - 2024
   * - ``get_type_overload``
     - 2.6
     - 2020
   * - ``call()``
     - 2.0
     - 2016
   * - ``.str()``
     - ?
     -
   * - ``.get_type()``
     - 2.6
     -
   * - ``==`` and ``!=``
     - 2.2
     - 2017
   * - ``.check()``
     - ?
     -
   * - ``object(handle, bool)``
     - ?
     -
   * - ``error_already_set.clear()``
     - 2.2
     - 2017
   * - ``obj.attr(…)`` as ``bool``
     - ?
     -
   * - ``.contains``
     - ? (maybe 2.4)
     -
   * - ``py::capsule`` two-argument with destructor
     - ?
     -



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
