.. _classes:

Object-oriented code
####################

Creating bindings for a custom type
===================================

Let's now look at a more complex example where we'll create bindings for a
custom C++ data structure named ``Pet``. Its definition is given below:

.. code-block:: cpp

    struct Pet {
        Pet(const std::string &name) : name(name) { }
        void setName(const std::string &name_) { name = name_; }
        const std::string &getName() const { return name; }

        std::string name;
    };

The binding code for ``Pet`` looks as follows:

.. code-block:: cpp

    #include <pybind11/pybind11.h>

    namespace py = pybind11;

    PYBIND11_MODULE(example, m) {
        py::class_<Pet>(m, "Pet")
            .def(py::init<const std::string &>())
            .def("setName", &Pet::setName)
            .def("getName", &Pet::getName);
    }

``py::class_`` creates bindings for a C++ *class* or *struct*-style data
structure. :func:`init` is a convenience function that takes the types of a
constructor's parameters as template arguments and wraps the corresponding
constructor (see the :ref:`custom_constructors` section for details).

.. note::

    Starting with pybind11v3, it is recommended to include `py::smart_holder`
    in most situations for safety, especially if you plan to support conversions
    to C++ smart pointers. See :ref:`smart_holder` for more information.

An interactive Python session demonstrating this example is shown below:

.. code-block:: pycon

    % python
    >>> import example
    >>> p = example.Pet("Molly")
    >>> print(p)
    <example.Pet object at 0x10cd98060>
    >>> p.getName()
    'Molly'
    >>> p.setName("Charly")
    >>> p.getName()
    'Charly'

.. seealso::

    Static member functions can be bound in the same way using
    :func:`class_::def_static`.

.. note::

    Binding C++ types in unnamed namespaces (also known as anonymous namespaces)
    works reliably on many platforms, but not all. The `XFAIL_CONDITION` in
    tests/test_unnamed_namespace_a.py encodes the currently known conditions.
    For background see `#4319 <https://github.com/pybind/pybind11/pull/4319>`_.
    If portability is a concern, it is therefore not recommended to bind C++
    types in unnamed namespaces. It will be safest to manually pick unique
    namespace names.

Keyword and default arguments
=============================
It is possible to specify keyword and default arguments using the syntax
discussed in the previous chapter. Refer to the sections :ref:`keyword_args`
and :ref:`default_args` for details.

Binding lambda functions
========================

Note how ``print(p)`` produced a rather useless summary of our data structure in the example above:

.. code-block:: pycon

    >>> print(p)
    <example.Pet object at 0x10cd98060>

To address this, we could bind a utility function that returns a human-readable
summary to the special method slot named ``__repr__``. Unfortunately, there is no
suitable functionality in the ``Pet`` data structure, and it would be nice if
we did not have to change it. This can easily be accomplished by binding a
Lambda function instead:

.. code-block:: cpp

        py::class_<Pet>(m, "Pet")
            .def(py::init<const std::string &>())
            .def("setName", &Pet::setName)
            .def("getName", &Pet::getName)
            .def("__repr__",
                [](const Pet &a) {
                    return "<example.Pet named '" + a.name + "'>";
                }
            );

Both stateless [#f1]_ and stateful lambda closures are supported by pybind11.
With the above change, the same Python code now produces the following output:

.. code-block:: pycon

    >>> print(p)
    <example.Pet named 'Molly'>

.. [#f1] Stateless closures are those with an empty pair of brackets ``[]`` as the capture object.

.. _properties:

Instance and static fields
==========================

We can also directly expose the ``name`` field using the
:func:`class_::def_readwrite` method. A similar :func:`class_::def_readonly`
method also exists for ``const`` fields.

.. code-block:: cpp

        py::class_<Pet>(m, "Pet")
            .def(py::init<const std::string &>())
            .def_readwrite("name", &Pet::name)
            // ... remainder ...

This makes it possible to write

.. code-block:: pycon

    >>> p = example.Pet("Molly")
    >>> p.name
    'Molly'
    >>> p.name = "Charly"
    >>> p.name
    'Charly'

Now suppose that ``Pet::name`` was a private internal variable
that can only be accessed via setters and getters.

.. code-block:: cpp

    class Pet {
    public:
        Pet(const std::string &name) : name(name) { }
        void setName(const std::string &name_) { name = name_; }
        const std::string &getName() const { return name; }
    private:
        std::string name;
    };

In this case, the method :func:`class_::def_property`
(:func:`class_::def_property_readonly` for read-only data) can be used to
provide a field-like interface within Python that will transparently call
the setter and getter functions:

.. code-block:: cpp

        py::class_<Pet>(m, "Pet")
            .def(py::init<const std::string &>())
            .def_property("name", &Pet::getName, &Pet::setName)
            // ... remainder ...

Write only properties can be defined by passing ``nullptr`` as the
input for the read function.

.. seealso::

    Similar functions :func:`class_::def_readwrite_static`,
    :func:`class_::def_readonly_static` :func:`class_::def_property_static`,
    and :func:`class_::def_property_readonly_static` are provided for binding
    static variables and properties. Please also see the section on
    :ref:`static_properties` in the advanced part of the documentation.

Dynamic attributes
==================

Native Python classes can pick up new attributes dynamically:

.. code-block:: pycon

    >>> class Pet:
    ...     name = "Molly"
    ...
    >>> p = Pet()
    >>> p.name = "Charly"  # overwrite existing
    >>> p.age = 2  # dynamically add a new attribute

By default, classes exported from C++ do not support this and the only writable
attributes are the ones explicitly defined using :func:`class_::def_readwrite`
or :func:`class_::def_property`.

.. code-block:: cpp

    py::class_<Pet>(m, "Pet")
        .def(py::init<>())
        .def_readwrite("name", &Pet::name);

Trying to set any other attribute results in an error:

.. code-block:: pycon

    >>> p = example.Pet()
    >>> p.name = "Charly"  # OK, attribute defined in C++
    >>> p.age = 2  # fail
    AttributeError: 'Pet' object has no attribute 'age'

To enable dynamic attributes for C++ classes, the :class:`py::dynamic_attr` tag
must be added to the :class:`py::class_` constructor:

.. code-block:: cpp

    py::class_<Pet>(m, "Pet", py::dynamic_attr())
        .def(py::init<>())
        .def_readwrite("name", &Pet::name);

Now everything works as expected:

.. code-block:: pycon

    >>> p = example.Pet()
    >>> p.name = "Charly"  # OK, overwrite value in C++
    >>> p.age = 2  # OK, dynamically add a new attribute
    >>> p.__dict__  # just like a native Python class
    {'age': 2}

Note that there is a small runtime cost for a class with dynamic attributes.
Not only because of the addition of a ``__dict__``, but also because of more
expensive garbage collection tracking which must be activated to resolve
possible circular references. Native Python classes incur this same cost by
default, so this is not anything to worry about. By default, pybind11 classes
are more efficient than native Python classes. Enabling dynamic attributes
just brings them on par.

.. _inheritance:

Inheritance and automatic downcasting
=====================================

Suppose now that the example consists of two data structures with an
inheritance relationship:

.. code-block:: cpp

    struct Pet {
        Pet(const std::string &name) : name(name) { }
        std::string name;
    };

    struct Dog : Pet {
        Dog(const std::string &name) : Pet(name) { }
        std::string bark() const { return "woof!"; }
    };

There are two different ways of indicating a hierarchical relationship to
pybind11: the first specifies the C++ base class as an extra template
parameter of the ``py::class_``:

.. code-block:: cpp

    py::class_<Pet>(m, "Pet")
       .def(py::init<const std::string &>())
       .def_readwrite("name", &Pet::name);

    // Method 1: template parameter:
    py::class_<Dog, Pet /* <- specify C++ parent type */>(m, "Dog")
        .def(py::init<const std::string &>())
        .def("bark", &Dog::bark);

Alternatively, we can also assign a name to the previously bound ``Pet``
``py::class_`` object and reference it when binding the ``Dog`` class:

.. code-block:: cpp

    py::class_<Pet> pet(m, "Pet");
    pet.def(py::init<const std::string &>())
       .def_readwrite("name", &Pet::name);

    // Method 2: pass parent class_ object:
    py::class_<Dog>(m, "Dog", pet /* <- specify Python parent type */)
        .def(py::init<const std::string &>())
        .def("bark", &Dog::bark);

Functionality-wise, both approaches are equivalent. Afterwards, instances will
expose fields and methods of both types:

.. code-block:: pycon

    >>> p = example.Dog("Molly")
    >>> p.name
    'Molly'
    >>> p.bark()
    'woof!'

The C++ classes defined above are regular non-polymorphic types with an
inheritance relationship. This is reflected in Python:

.. code-block:: cpp

    // Return a base pointer to a derived instance
    m.def("pet_store", []() { return std::unique_ptr<Pet>(new Dog("Molly")); });

.. code-block:: pycon

    >>> p = example.pet_store()
    >>> type(p)  # `Dog` instance behind `Pet` pointer
    Pet          # no pointer downcasting for regular non-polymorphic types
    >>> p.bark()
    AttributeError: 'Pet' object has no attribute 'bark'

The function returned a ``Dog`` instance, but because it's a non-polymorphic
type behind a base pointer, Python only sees a ``Pet``. In C++, a type is only
considered polymorphic if it has at least one virtual function and pybind11
will automatically recognize this:

.. code-block:: cpp

    struct PolymorphicPet {
        virtual ~PolymorphicPet() = default;
    };

    struct PolymorphicDog : PolymorphicPet {
        std::string bark() const { return "woof!"; }
    };

    // Same binding code
    py::class_<PolymorphicPet>(m, "PolymorphicPet");
    py::class_<PolymorphicDog, PolymorphicPet>(m, "PolymorphicDog")
        .def(py::init<>())
        .def("bark", &PolymorphicDog::bark);

    // Again, return a base pointer to a derived instance
    m.def("pet_store2", []() { return std::unique_ptr<PolymorphicPet>(new PolymorphicDog); });

.. code-block:: pycon

    >>> p = example.pet_store2()
    >>> type(p)
    PolymorphicDog  # automatically downcast
    >>> p.bark()
    'woof!'

Given a pointer to a polymorphic base, pybind11 performs automatic downcasting
to the actual derived type. Note that this goes beyond the usual situation in
C++: we don't just get access to the virtual functions of the base, we get the
concrete derived type including functions and attributes that the base type may
not even be aware of.

.. seealso::

    For more information about polymorphic behavior see :ref:`overriding_virtuals`.


Overloaded methods
==================

Sometimes there are several overloaded C++ methods with the same name taking
different kinds of input arguments:

.. code-block:: cpp

    struct Pet {
        Pet(const std::string &name, int age) : name(name), age(age) { }

        void set(int age_) { age = age_; }
        void set(const std::string &name_) { name = name_; }

        std::string name;
        int age;
    };

Attempting to bind ``Pet::set`` will cause an error since the compiler does not
know which method the user intended to select. We can disambiguate by casting
them to function pointers. Binding multiple functions to the same Python name
automatically creates a chain of function overloads that will be tried in
sequence.

.. code-block:: cpp

    py::class_<Pet>(m, "Pet")
       .def(py::init<const std::string &, int>())
       .def("set", static_cast<void (Pet::*)(int)>(&Pet::set), "Set the pet's age")
       .def("set", static_cast<void (Pet::*)(const std::string &)>(&Pet::set), "Set the pet's name");

The overload signatures are also visible in the method's docstring:

.. code-block:: pycon

    >>> help(example.Pet)

    class Pet(__builtin__.object)
     |  Methods defined here:
     |
     |  __init__(...)
     |      Signature : (Pet, str, int) -> NoneType
     |
     |  set(...)
     |      1. Signature : (Pet, int) -> NoneType
     |
     |      Set the pet's age
     |
     |      2. Signature : (Pet, str) -> NoneType
     |
     |      Set the pet's name

If you have a C++14 compatible compiler [#cpp14]_, you can use an alternative
syntax to cast the overloaded function:

.. code-block:: cpp

    py::class_<Pet>(m, "Pet")
        .def("set", py::overload_cast<int>(&Pet::set), "Set the pet's age")
        .def("set", py::overload_cast<const std::string &>(&Pet::set), "Set the pet's name");

Here, ``py::overload_cast`` only requires the parameter types to be specified.
The return type and class are deduced. This avoids the additional noise of
``void (Pet::*)()`` as seen in the raw cast. If a function is overloaded based
on constness, the ``py::const_`` tag should be used:

.. code-block:: cpp

    struct Widget {
        int foo(int x, float y);
        int foo(int x, float y) const;
    };

    py::class_<Widget>(m, "Widget")
       .def("foo_mutable", py::overload_cast<int, float>(&Widget::foo))
       .def("foo_const",   py::overload_cast<int, float>(&Widget::foo, py::const_));

If you prefer the ``py::overload_cast`` syntax but have a C++11 compatible compiler only,
you can use ``py::detail::overload_cast_impl`` with an additional set of parentheses:

.. code-block:: cpp

    template <typename... Args>
    using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

    py::class_<Pet>(m, "Pet")
        .def("set", overload_cast_<int>()(&Pet::set), "Set the pet's age")
        .def("set", overload_cast_<const std::string &>()(&Pet::set), "Set the pet's name");

.. [#cpp14] A compiler which supports the ``-std=c++14`` flag.

.. note::

    To define multiple overloaded constructors, simply declare one after the
    other using the ``.def(py::init<...>())`` syntax. The existing machinery
    for specifying keyword and default arguments also works.

☝️ Pitfalls with raw pointers and shared ownership
==================================================

``py::class_``-wrapped objects automatically manage the lifetime of the
wrapped C++ object, in collaboration with the chosen holder type (see
:ref:`py_class_holder`). When wrapping C++ functions involving raw pointers,
care needs to be taken to not accidentally undermine automatic lifetime
management. For example, ownership is inadvertently transferred here:

.. code-block:: cpp

    class Child { };

    class Parent {
    public:
       Parent() : child(std::make_shared<Child>()) { }
       Child *get_child() { return child.get(); }  /* DANGER */
    private:
        std::shared_ptr<Child> child;
    };

    PYBIND11_MODULE(example, m) {
        py::class_<Child, std::shared_ptr<Child>>(m, "Child");

        py::class_<Parent, std::shared_ptr<Parent>>(m, "Parent")
           .def(py::init<>())
           .def("get_child", &Parent::get_child);  /* PROBLEM */
    }

The following Python code leads to undefined behavior, likely resulting in
a segmentation fault.

.. code-block:: python

   from example import Parent

   print(Parent().get_child())

Part of the ``/* PROBLEM */`` here is that pybind11 falls back to using
``return_value_policy::take_ownership`` as the default (see
:ref:`return_value_policies`). The fact that the ``Child`` instance is
already managed by ``std::shared_ptr<Child>`` is lost. Therefore pybind11
will create a second independent ``std::shared_ptr<Child>`` that also
claims ownership of the pointer, eventually leading to heap-use-after-free
or double-free errors.

There are various ways to resolve this issue, either by changing
the ``Child`` or ``Parent`` C++ implementations (e.g. using
``std::enable_shared_from_this<Child>`` as a base class for
``Child``, or adding a member function to ``Parent`` that returns
``std::shared_ptr<Child>``), or if that is not feasible, by using
``return_value_policy::reference_internal``. What is the best approach
depends on the exact situation.

A highly effective way to stay in the clear — even in pure C++, but
especially when binding C++ code to Python — is to consistently prefer
``std::shared_ptr`` or ``std::unique_ptr`` over passing raw pointers.

.. _native_enum:

Enumerations and internal types
===============================

Let's now suppose that the example class contains internal types like enumerations, e.g.:

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

    #include <pybind11/native_enum.h> // Not already included with pybind11.h

    py::class_<Pet> pet(m, "Pet");

    pet.def(py::init<const std::string &, Pet::Kind>())
        .def_readwrite("name", &Pet::name)
        .def_readwrite("type", &Pet::type)
        .def_readwrite("attr", &Pet::attr);

    py::native_enum<Pet::Kind>(pet, "Kind", "enum.Enum")
        .value("Dog", Pet::Kind::Dog)
        .value("Cat", Pet::Kind::Cat)
        .export_values()
        .finalize();

    py::class_<Pet::Attributes>(pet, "Attributes")
        .def(py::init<>())
        .def_readwrite("age", &Pet::Attributes::age);


To ensure that the nested types ``Kind`` and ``Attributes`` are created
within the scope of ``Pet``, the ``pet`` ``py::class_`` instance must be
supplied to the ``py::native_enum`` and ``py::class_`` constructors. The
``.export_values()`` function is available for exporting the enum entries
into the parent scope, if desired.

``py::native_enum`` was introduced with pybind11v3. It binds C++ enum types
to native Python enum types, typically types in Python's
`stdlib enum <https://docs.python.org/3/library/enum.html>`_ module,
which are `PEP 435 compatible <https://peps.python.org/pep-0435/>`_.
This is the recommended way to bind C++ enums.
The older ``py::enum_`` is not PEP 435 compatible
(see `issue #2332 <https://github.com/pybind/pybind11/issues/2332>`_)
but remains supported indefinitely for backward compatibility.
New bindings should prefer ``py::native_enum``.

.. note::

    The deprecated ``py::enum_`` is :ref:`documented here <deprecated_enum>`.

The ``.finalize()`` call above is needed because Python's native enums
cannot be built incrementally — all name/value pairs need to be passed at
once. To achieve this, ``py::native_enum`` acts as a buffer to collect the
name/value pairs. The ``.finalize()`` call uses the accumulated name/value
pairs to build the arguments for constructing a native Python enum type.

The ``py::native_enum`` constructor takes a third argument,
``native_type_name``, which specifies the fully qualified name of the Python
base class to use — e.g., ``"enum.Enum"`` or ``"enum.IntEnum"``. A fourth
optional argument, ``class_doc``, provides the docstring for the generated
class.

For example:

.. code-block:: cpp

    py::native_enum<Pet::Kind>(pet, "Kind", "enum.IntEnum", "Constant specifying the kind of pet")

You may use any fully qualified Python name for ``native_type_name``.
The only requirement is that the named type is similar to
`enum.Enum <https://docs.python.org/3/library/enum.html#enum.Enum>`_
in these ways:

* Has a `constructor similar to that of enum.Enum
  <https://docs.python.org/3/howto/enum.html#functional-api>`_::

    Colors = enum.Enum("Colors", (("Red", 0), ("Green", 1)))

* A `C++ underlying <https://en.cppreference.com/w/cpp/types/underlying_type>`_
  enum value can be passed to the constructor for the Python enum value::

    red = Colors(0)

* The enum values have a ``.value`` property yielding a value that
  can be cast to the C++ underlying type::

    underlying = red.value

As of Python 3.13, the compatible `types in the stdlib enum module
<https://docs.python.org/3/library/enum.html#module-contents>`_ are:
``Enum``, ``IntEnum``, ``Flag``, ``IntFlag``.

.. note::

    In rare cases, a C++ enum may be bound to Python via a
    :ref:`custom type caster <custom_type_caster>`. In such cases, a
    template specialization like this may be required:

    .. code-block:: cpp

        #if defined(PYBIND11_HAS_NATIVE_ENUM)
        namespace pybind11::detail {
        template <typename FancyEnum>
        struct type_caster_enum_type_enabled<
            FancyEnum,
            enable_if_t<is_fancy_enum<FancyEnum>::value>> : std::false_type {};
        }
        #endif

    This specialization is needed only if the custom type caster is templated.

    The ``PYBIND11_HAS_NATIVE_ENUM`` guard is needed only if backward
    compatibility with pybind11v2 is required.
