Deprecated
##########

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
