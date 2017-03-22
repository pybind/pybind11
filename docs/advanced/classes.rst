Classes
#######

This section presents advanced binding code for classes and it is assumed
that you are already familiar with the basics from :doc:`/classes`.

.. _overriding_virtuals:

Overriding virtual functions in Python
======================================

Suppose that a C++ class or interface has a virtual function that we'd like to
to override from within Python (we'll focus on the class ``Animal``; ``Dog`` is
given as a specific example of how one would do this with traditional C++
code).

.. code-block:: cpp

    class Animal {
    public:
        virtual ~Animal() { }
        virtual std::string go(int n_times) = 0;
    };

    class Dog : public Animal {
    public:
        std::string go(int n_times) override {
            std::string result;
            for (int i=0; i<n_times; ++i)
                result += "woof! ";
            return result;
        }
    };

Let's also suppose that we are given a plain function which calls the
function ``go()`` on an arbitrary ``Animal`` instance.

.. code-block:: cpp

    std::string call_go(Animal *animal) {
        return animal->go(3);
    }

Normally, the binding code for these classes would look as follows:

.. code-block:: cpp

    PYBIND11_PLUGIN(example) {
        py::module m("example", "pybind11 example plugin");

        py::class_<Animal> animal(m, "Animal");
        animal
            .def("go", &Animal::go);

        py::class_<Dog>(m, "Dog", animal)
            .def(py::init<>());

        m.def("call_go", &call_go);

        return m.ptr();
    }

However, these bindings are impossible to extend: ``Animal`` is not
constructible, and we clearly require some kind of "trampoline" that
redirects virtual calls back to Python.

Defining a new type of ``Animal`` from within Python is possible but requires a
helper class that is defined as follows:

.. code-block:: cpp

    class PyAnimal : public Animal {
    public:
        /* Inherit the constructors */
        using Animal::Animal;

        /* Trampoline (need one for each virtual function) */
        std::string go(int n_times) override {
            PYBIND11_OVERLOAD_PURE(
                std::string, /* Return type */
                Animal,      /* Parent class */
                go,          /* Name of function in C++ (must match Python name) */
                n_times      /* Argument(s) */
            );
        }
    };

The macro :func:`PYBIND11_OVERLOAD_PURE` should be used for pure virtual
functions, and :func:`PYBIND11_OVERLOAD` should be used for functions which have
a default implementation.  There are also two alternate macros
:func:`PYBIND11_OVERLOAD_PURE_NAME` and :func:`PYBIND11_OVERLOAD_NAME` which
take a string-valued name argument between the *Parent class* and *Name of the
function* slots, which defines the name of function in Python. This is required 
when the C++ and Python versions of the
function have different names, e.g.  ``operator()`` vs ``__call__``.

The binding code also needs a few minor adaptations (highlighted):

.. code-block:: cpp
    :emphasize-lines: 4,6,7

    PYBIND11_PLUGIN(example) {
        py::module m("example", "pybind11 example plugin");

        py::class_<Animal, PyAnimal /* <--- trampoline*/> animal(m, "Animal");
        animal
            .def(py::init<>())
            .def("go", &Animal::go);

        py::class_<Dog>(m, "Dog", animal)
            .def(py::init<>());

        m.def("call_go", &call_go);

        return m.ptr();
    }

Importantly, pybind11 is made aware of the trampoline helper class by
specifying it as an extra template argument to :class:`class_`. (This can also
be combined with other template arguments such as a custom holder type; the
order of template types does not matter).  Following this, we are able to
define a constructor as usual.

Bindings should be made against the actual class, not the trampoline helper class.

.. code-block:: cpp

    py::class_<Animal, PyAnimal /* <--- trampoline*/> animal(m, "Animal");
        animal
            .def(py::init<>())
            .def("go", &PyAnimal::go); /* <--- THIS IS WRONG, use &Animal::go */

Note, however, that the above is sufficient for allowing python classes to
extend ``Animal``, but not ``Dog``: see ref:`virtual_and_inheritance` for the
necessary steps required to providing proper overload support for inherited
classes.

The Python session below shows how to override ``Animal::go`` and invoke it via
a virtual method call.

.. code-block:: pycon

    >>> from example import *
    >>> d = Dog()
    >>> call_go(d)
    u'woof! woof! woof! '
    >>> class Cat(Animal):
    ...     def go(self, n_times):
    ...             return "meow! " * n_times
    ...
    >>> c = Cat()
    >>> call_go(c)
    u'meow! meow! meow! '

Please take a look at the :ref:`macro_notes` before using this feature.

.. note::

    When the overridden type returns a reference or pointer to a type that
    pybind11 converts from Python (for example, numeric values, std::string,
    and other built-in value-converting types), there are some limitations to
    be aware of:

    - because in these cases there is no C++ variable to reference (the value
      is stored in the referenced Python variable), pybind11 provides one in
      the PYBIND11_OVERLOAD macros (when needed) with static storage duration.
      Note that this means that invoking the overloaded method on *any*
      instance will change the referenced value stored in *all* instances of
      that type.

    - Attempts to modify a non-const reference will not have the desired
      effect: it will change only the static cache variable, but this change
      will not propagate to underlying Python instance, and the change will be
      replaced the next time the overload is invoked.

.. seealso::

    The file :file:`tests/test_virtual_functions.cpp` contains a complete
    example that demonstrates how to override virtual functions using pybind11
    in more detail.

.. _virtual_and_inheritance:

Combining virtual functions and inheritance
===========================================

When combining virtual methods with inheritance, you need to be sure to provide
an override for each method for which you want to allow overrides from derived
python classes.  For example, suppose we extend the above ``Animal``/``Dog``
example as follows:

.. code-block:: cpp

    class Animal {
    public:
        virtual std::string go(int n_times) = 0;
        virtual std::string name() { return "unknown"; }
    };
    class Dog : public Animal {
    public:
        std::string go(int n_times) override {
            std::string result;
            for (int i=0; i<n_times; ++i)
                result += bark() + " ";
            return result;
        }
        virtual std::string bark() { return "woof!"; }
    };

then the trampoline class for ``Animal`` must, as described in the previous
section, override ``go()`` and ``name()``, but in order to allow python code to
inherit properly from ``Dog``, we also need a trampoline class for ``Dog`` that
overrides both the added ``bark()`` method *and* the ``go()`` and ``name()``
methods inherited from ``Animal`` (even though ``Dog`` doesn't directly
override the ``name()`` method):

.. code-block:: cpp

    class PyAnimal : public Animal {
    public:
        using Animal::Animal; // Inherit constructors
        std::string go(int n_times) override { PYBIND11_OVERLOAD_PURE(std::string, Animal, go, n_times); }
        std::string name() override { PYBIND11_OVERLOAD(std::string, Animal, name, ); }
    };
    class PyDog : public Dog {
    public:
        using Dog::Dog; // Inherit constructors
        std::string go(int n_times) override { PYBIND11_OVERLOAD_PURE(std::string, Dog, go, n_times); }
        std::string name() override { PYBIND11_OVERLOAD(std::string, Dog, name, ); }
        std::string bark() override { PYBIND11_OVERLOAD(std::string, Dog, bark, ); }
    };

.. note::

    Note the trailing commas in the ``PYBIND11_OVERLOAD`` calls to ``name()``
    and ``bark()``. These are needed to portably implement a trampoline for a
    function that does not take any arguments. For functions that take
    a nonzero number of arguments, the trailing comma must be omitted.

A registered class derived from a pybind11-registered class with virtual
methods requires a similar trampoline class, *even if* it doesn't explicitly
declare or override any virtual methods itself:

.. code-block:: cpp

    class Husky : public Dog {};
    class PyHusky : public Husky {
    public:
        using Husky::Husky; // Inherit constructors
        std::string go(int n_times) override { PYBIND11_OVERLOAD_PURE(std::string, Husky, go, n_times); }
        std::string name() override { PYBIND11_OVERLOAD(std::string, Husky, name, ); }
        std::string bark() override { PYBIND11_OVERLOAD(std::string, Husky, bark, ); }
    };

There is, however, a technique that can be used to avoid this duplication
(which can be especially helpful for a base class with several virtual
methods).  The technique involves using template trampoline classes, as
follows:

.. code-block:: cpp

    template <class AnimalBase = Animal> class PyAnimal : public AnimalBase {
    public:
        using AnimalBase::AnimalBase; // Inherit constructors
        std::string go(int n_times) override { PYBIND11_OVERLOAD_PURE(std::string, AnimalBase, go, n_times); }
        std::string name() override { PYBIND11_OVERLOAD(std::string, AnimalBase, name, ); }
    };
    template <class DogBase = Dog> class PyDog : public PyAnimal<DogBase> {
    public:
        using PyAnimal<DogBase>::PyAnimal; // Inherit constructors
        // Override PyAnimal's pure virtual go() with a non-pure one:
        std::string go(int n_times) override { PYBIND11_OVERLOAD(std::string, DogBase, go, n_times); }
        std::string bark() override { PYBIND11_OVERLOAD(std::string, DogBase, bark, ); }
    };

This technique has the advantage of requiring just one trampoline method to be
declared per virtual method and pure virtual method override.  It does,
however, require the compiler to generate at least as many methods (and
possibly more, if both pure virtual and overridden pure virtual methods are
exposed, as above).

The classes are then registered with pybind11 using:

.. code-block:: cpp

    py::class_<Animal, PyAnimal<>> animal(m, "Animal");
    py::class_<Dog, PyDog<>> dog(m, "Dog");
    py::class_<Husky, PyDog<Husky>> husky(m, "Husky");
    // ... add animal, dog, husky definitions

Note that ``Husky`` did not require a dedicated trampoline template class at
all, since it neither declares any new virtual methods nor provides any pure
virtual method implementations.

With either the repeated-virtuals or templated trampoline methods in place, you
can now create a python class that inherits from ``Dog``:

.. code-block:: python

    class ShihTzu(Dog):
        def bark(self):
            return "yip!"

.. seealso::

    See the file :file:`tests/test_virtual_functions.cpp` for complete examples
    using both the duplication and templated trampoline approaches.

Extended trampoline class functionality
=======================================

The trampoline classes described in the previous sections are, by default, only
initialized when needed.  More specifically, they are initialized when a python
class actually inherits from a registered type (instead of merely creating an
instance of the registered type), or when a registered constructor is only
valid for the trampoline class but not the registered class.  This is primarily
for performance reasons: when the trampoline class is not needed for anything
except virtual method dispatching, not initializing the trampoline class
improves performance by avoiding needing to do a run-time check to see if the
inheriting python instance has an overloaded method.

Sometimes, however, it is useful to always initialize a trampoline class as an
intermediate class that does more than just handle virtual method dispatching.
For example, such a class might perform extra class initialization, extra
destruction operations, and might define new members and methods to enable a
more python-like interface to a class.

In order to tell pybind11 that it should *always* initialize the trampoline
class when creating new instances of a type, the class constructors should be
declared using ``py::init_alias<Args, ...>()`` instead of the usual
``py::init<Args, ...>()``.  This forces construction via the trampoline class,
ensuring member initialization and (eventual) destruction.

.. seealso::

    See the file :file:`tests/test_alias_initialization.cpp` for complete examples
    showing both normal and forced trampoline instantiation.

.. _custom_constructors:

Custom constructors
===================

The syntax for binding constructors was previously introduced, but it only
works when a constructor with the given parameters actually exists on the C++
side. To extend this to more general cases, let's take a look at what actually
happens under the hood: the following statement

.. code-block:: cpp

    py::class_<Example>(m, "Example")
        .def(py::init<int>());

is short hand notation for

.. code-block:: cpp

    py::class_<Example>(m, "Example")
        .def("__init__",
            [](Example &instance, int arg) {
                new (&instance) Example(arg);
            }
        );

In other words, :func:`init` creates an anonymous function that invokes an
in-place constructor. Memory allocation etc. is already take care of beforehand
within pybind11.

.. _classes_with_non_public_destructors:

Non-public destructors
======================

If a class has a private or protected destructor (as might e.g. be the case in
a singleton pattern), a compile error will occur when creating bindings via
pybind11. The underlying issue is that the ``std::unique_ptr`` holder type that
is responsible for managing the lifetime of instances will reference the
destructor even if no deallocations ever take place. In order to expose classes
with private or protected destructors, it is possible to override the holder
type via a holder type argument to ``class_``. Pybind11 provides a helper class
``py::nodelete`` that disables any destructor invocations. In this case, it is
crucial that instances are deallocated on the C++ side to avoid memory leaks.

.. code-block:: cpp

    /* ... definition ... */

    class MyClass {
    private:
        ~MyClass() { }
    };

    /* ... binding code ... */

    py::class_<MyClass, std::unique_ptr<MyClass, py::nodelete>>(m, "MyClass")
        .def(py::init<>())

.. _implicit_conversions:

Implicit conversions
====================

Suppose that instances of two types ``A`` and ``B`` are used in a project, and
that an ``A`` can easily be converted into an instance of type ``B`` (examples of this
could be a fixed and an arbitrary precision number type).

.. code-block:: cpp

    py::class_<A>(m, "A")
        /// ... members ...

    py::class_<B>(m, "B")
        .def(py::init<A>())
        /// ... members ...

    m.def("func",
        [](const B &) { /* .... */ }
    );

To invoke the function ``func`` using a variable ``a`` containing an ``A``
instance, we'd have to write ``func(B(a))`` in Python. On the other hand, C++
will automatically apply an implicit type conversion, which makes it possible
to directly write ``func(a)``.

In this situation (i.e. where ``B`` has a constructor that converts from
``A``), the following statement enables similar implicit conversions on the
Python side:

.. code-block:: cpp

    py::implicitly_convertible<A, B>();

.. note::

    Implicit conversions from ``A`` to ``B`` only work when ``B`` is a custom
    data type that is exposed to Python via pybind11.

.. _static_properties:

Static properties
=================

The section on :ref:`properties` discussed the creation of instance properties
that are implemented in terms of C++ getters and setters.

Static properties can also be created in a similar way to expose getters and
setters of static class attributes. Note that the implicit ``self`` argument
also exists in this case and is used to pass the Python ``type`` subclass
instance. This parameter will often not be needed by the C++ side, and the
following example illustrates how to instantiate a lambda getter function
that ignores it:

.. code-block:: cpp

    py::class_<Foo>(m, "Foo")
        .def_property_readonly_static("foo", [](py::object /* self */) { return Foo(); });

Operator overloading
====================

Suppose that we're given the following ``Vector2`` class with a vector addition
and scalar multiplication operation, all implemented using overloaded operators
in C++.

.. code-block:: cpp

    class Vector2 {
    public:
        Vector2(float x, float y) : x(x), y(y) { }

        Vector2 operator+(const Vector2 &v) const { return Vector2(x + v.x, y + v.y); }
        Vector2 operator*(float value) const { return Vector2(x * value, y * value); }
        Vector2& operator+=(const Vector2 &v) { x += v.x; y += v.y; return *this; }
        Vector2& operator*=(float v) { x *= v; y *= v; return *this; }

        friend Vector2 operator*(float f, const Vector2 &v) {
            return Vector2(f * v.x, f * v.y);
        }

        std::string toString() const {
            return "[" + std::to_string(x) + ", " + std::to_string(y) + "]";
        }
    private:
        float x, y;
    };

The following snippet shows how the above operators can be conveniently exposed
to Python.

.. code-block:: cpp

    #include <pybind11/operators.h>

    PYBIND11_PLUGIN(example) {
        py::module m("example", "pybind11 example plugin");

        py::class_<Vector2>(m, "Vector2")
            .def(py::init<float, float>())
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def(py::self *= float())
            .def(float() * py::self)
            .def(py::self * float())
            .def("__repr__", &Vector2::toString);

        return m.ptr();
    }

Note that a line like

.. code-block:: cpp

            .def(py::self * float())

is really just short hand notation for

.. code-block:: cpp

    .def("__mul__", [](const Vector2 &a, float b) {
        return a * b;
    }, py::is_operator())

This can be useful for exposing additional operators that don't exist on the
C++ side, or to perform other types of customization. The ``py::is_operator``
flag marker is needed to inform pybind11 that this is an operator, which
returns ``NotImplemented`` when invoked with incompatible arguments rather than
throwing a type error.

.. note::

    To use the more convenient ``py::self`` notation, the additional
    header file :file:`pybind11/operators.h` must be included.

.. seealso::

    The file :file:`tests/test_operator_overloading.cpp` contains a
    complete example that demonstrates how to work with overloaded operators in
    more detail.

Pickling support
================

Python's ``pickle`` module provides a powerful facility to serialize and
de-serialize a Python object graph into a binary data stream. To pickle and
unpickle C++ classes using pybind11, two additional functions must be provided.
Suppose the class in question has the following signature:

.. code-block:: cpp

    class Pickleable {
    public:
        Pickleable(const std::string &value) : m_value(value) { }
        const std::string &value() const { return m_value; }

        void setExtra(int extra) { m_extra = extra; }
        int extra() const { return m_extra; }
    private:
        std::string m_value;
        int m_extra = 0;
    };

The binding code including the requisite ``__setstate__`` and ``__getstate__`` methods [#f3]_
looks as follows:

.. code-block:: cpp

    py::class_<Pickleable>(m, "Pickleable")
        .def(py::init<std::string>())
        .def("value", &Pickleable::value)
        .def("extra", &Pickleable::extra)
        .def("setExtra", &Pickleable::setExtra)
        .def("__getstate__", [](const Pickleable &p) {
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.value(), p.extra());
        })
        .def("__setstate__", [](Pickleable &p, py::tuple t) {
            if (t.size() != 2)
                throw std::runtime_error("Invalid state!");

            /* Invoke the in-place constructor. Note that this is needed even
               when the object just has a trivial default constructor */
            new (&p) Pickleable(t[0].cast<std::string>());

            /* Assign any additional state */
            p.setExtra(t[1].cast<int>());
        });

An instance can now be pickled as follows:

.. code-block:: python

    try:
        import cPickle as pickle  # Use cPickle on Python 2.7
    except ImportError:
        import pickle

    p = Pickleable("test_value")
    p.setExtra(15)
    data = pickle.dumps(p, 2)

Note that only the cPickle module is supported on Python 2.7. The second
argument to ``dumps`` is also crucial: it selects the pickle protocol version
2, since the older version 1 is not supported. Newer versions are also fine—for
instance, specify ``-1`` to always use the latest available version. Beware:
failure to follow these instructions will cause important pybind11 memory
allocation routines to be skipped during unpickling, which will likely lead to
memory corruption and/or segmentation faults.

.. seealso::

    The file :file:`tests/test_pickling.cpp` contains a complete example
    that demonstrates how to pickle and unpickle types using pybind11 in more
    detail.

.. [#f3] http://docs.python.org/3/library/pickle.html#pickling-class-instances

Multiple Inheritance
====================

pybind11 can create bindings for types that derive from multiple base types
(aka. *multiple inheritance*). To do so, specify all bases in the template
arguments of the ``class_`` declaration:

.. code-block:: cpp

    py::class_<MyType, BaseType1, BaseType2, BaseType3>(m, "MyType")
       ...

The base types can be specified in arbitrary order, and they can even be
interspersed with alias types and holder types (discussed earlier in this
document)---pybind11 will automatically find out which is which. The only
requirement is that the first template argument is the type to be declared.

There are two caveats regarding the implementation of this feature:

1. When only one base type is specified for a C++ type that actually has
   multiple bases, pybind11 will assume that it does not participate in
   multiple inheritance, which can lead to undefined behavior. In such cases,
   add the tag ``multiple_inheritance``:

    .. code-block:: cpp

        py::class_<MyType, BaseType2>(m, "MyType", py::multiple_inheritance());

   The tag is redundant and does not need to be specified when multiple base
   types are listed.

2. As was previously discussed in the section on :ref:`overriding_virtuals`, it
   is easy to create Python types that derive from C++ classes. It is even
   possible to make use of multiple inheritance to declare a Python class which
   has e.g. a C++ and a Python class as bases. However, any attempt to create a
   type that has *two or more* C++ classes in its hierarchy of base types will
   fail with a fatal error message: ``TypeError: multiple bases have instance
   lay-out conflict``. Core Python types that are implemented in C (e.g.
   ``dict``, ``list``, ``Exception``, etc.) also fall under this combination
   and cannot be combined with C++ types bound using pybind11 via multiple
   inheritance.
