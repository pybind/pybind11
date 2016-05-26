.. _advanced:

Advanced topics
###############

For brevity, the rest of this chapter assumes that the following two lines are
present:

.. code-block:: cpp

    #include <pybind11/pybind11.h>

    namespace py = pybind11;

Exporting constants and mutable objects
=======================================

To expose a C++ constant, use the ``attr`` function to register it in a module
as shown below. The ``int_`` class is one of many small wrapper objects defined
in ``pybind11/pytypes.h``. General objects (including integers) can also be
converted using the function ``cast``.

.. code-block:: cpp

    PYBIND11_PLUGIN(example) {
        py::module m("example", "pybind11 example plugin");
        m.attr("MY_CONSTANT") = py::int_(123);
        m.attr("MY_CONSTANT_2") = py::cast(new MyObject());
    }

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
    })

This can be useful for exposing additional operators that don't exist on the
C++ side, or to perform other types of customization.

.. note::

    To use the more convenient ``py::self`` notation, the additional
    header file :file:`pybind11/operators.h` must be included.

.. seealso::

    The file :file:`example/example3.cpp` contains a complete example that
    demonstrates how to work with overloaded operators in more detail.

Callbacks and passing anonymous functions
=========================================

The C++11 standard brought lambda functions and the generic polymorphic
function wrapper ``std::function<>`` to the C++ programming language, which
enable powerful new ways of working with functions. Lambda functions come in
two flavors: stateless lambda function resemble classic function pointers that
link to an anonymous piece of code, while stateful lambda functions
additionally depend on captured variables that are stored in an anonymous
*lambda closure object*.

Here is a simple example of a C++ function that takes an arbitrary function
(stateful or stateless) with signature ``int -> int`` as an argument and runs
it with the value 10.

.. code-block:: cpp

    int func_arg(const std::function<int(int)> &f) {
        return f(10);
    }

The example below is more involved: it takes a function of signature ``int -> int``
and returns another function of the same kind. The return value is a stateful
lambda function, which stores the value ``f`` in the capture object and adds 1 to
its return value upon execution.

.. code-block:: cpp

    std::function<int(int)> func_ret(const std::function<int(int)> &f) {
        return [f](int i) {
            return f(i) + 1;
        };
    }

After including the extra header file :file:`pybind11/functional.h`, it is almost
trivial to generate binding code for both of these functions.

.. code-block:: cpp

    #include <pybind11/functional.h>

    PYBIND11_PLUGIN(example) {
        py::module m("example", "pybind11 example plugin");

        m.def("func_arg", &func_arg);
        m.def("func_ret", &func_ret);

        return m.ptr();
    }

The following interactive session shows how to call them from Python.

.. code-block:: python

    $ python
    >>> import example
    >>> def square(i):
    ...     return i * i
    ...
    >>> example.func_arg(square)
    100L
    >>> square_plus_1 = example.func_ret(square)
    >>> square_plus_1(4)
    17L
    >>>

.. note::

    This functionality is very useful when generating bindings for callbacks in
    C++ libraries (e.g. a graphical user interface library).

    The file :file:`example/example5.cpp` contains a complete example that
    demonstrates how to work with callbacks and anonymous functions in more detail.

.. warning::

    Keep in mind that passing a function from C++ to Python (or vice versa)
    will instantiate a piece of wrapper code that translates function
    invocations between the two languages. Copying the same function back and
    forth between Python and C++ many times in a row will cause these wrappers
    to accumulate, which can decrease performance.

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
        std::string go(int n_times) {
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
        std::string go(int n_times) {
            PYBIND11_OVERLOAD_PURE(
                std::string, /* Return type */
                Animal,      /* Parent class */
                go,          /* Name of function */
                n_times      /* Argument(s) */
            );
        }
    };

The macro :func:`PYBIND11_OVERLOAD_PURE` should be used for pure virtual
functions, and :func:`PYBIND11_OVERLOAD` should be used for functions which have
a default implementation. 

There are also two alternate macros :func:`PYBIND11_OVERLOAD_PURE_NAME` and
:func:`PYBIND11_OVERLOAD_NAME` which take a string-valued name argument
after the *Name of the function* slot. This is useful when the C++ and Python
versions of the function have different names, e.g. ``operator()`` vs ``__call__``.

The binding code also needs a few minor adaptations (highlighted):

.. code-block:: cpp
    :emphasize-lines: 4,6,7

    PYBIND11_PLUGIN(example) {
        py::module m("example", "pybind11 example plugin");

        py::class_<Animal, std::unique_ptr<Animal>, PyAnimal /* <--- trampoline*/> animal(m, "Animal");
        animal
            .def(py::init<>())
            .def("go", &Animal::go);

        py::class_<Dog>(m, "Dog", animal)
            .def(py::init<>());

        m.def("call_go", &call_go);

        return m.ptr();
    }

Importantly, pybind11 is made aware of the trampoline trampoline helper class
by specifying it as the *third* template argument to :class:`class_`. The
second argument with the unique pointer is simply the default holder type used
by pybind11. Following this, we are able to define a constructor as usual.

The Python session below shows how to override ``Animal::go`` and invoke it via
a virtual method call.

.. code-block:: python

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

.. warning::

    The :func:`PYBIND11_OVERLOAD_*` calls are all just macros, which means that
    they can get confused by commas in a template argument such as
    ``PYBIND11_OVERLOAD(MyReturnValue<T1, T2>, myFunc)``. In this case, the
    preprocessor assumes that the comma indicates the beginnning of the next
    parameter. Use a ``typedef`` to bind the template to another name and use
    it in the macro to avoid this problem.

.. seealso::

    The file :file:`example/example12.cpp` contains a complete example that
    demonstrates how to override virtual functions using pybind11 in more
    detail.


Global Interpreter Lock (GIL)
=============================

The classes :class:`gil_scoped_release` and :class:`gil_scoped_acquire` can be
used to acquire and release the global interpreter lock in the body of a C++
function call. In this way, long-running C++ code can be parallelized using
multiple Python threads. Taking the previous section as an example, this could
be realized as follows (important changes highlighted):

.. code-block:: cpp
    :emphasize-lines: 8,9,33,34

    class PyAnimal : public Animal {
    public:
        /* Inherit the constructors */
        using Animal::Animal;

        /* Trampoline (need one for each virtual function) */
        std::string go(int n_times) {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERLOAD_PURE(
                std::string, /* Return type */
                Animal,      /* Parent class */
                go,          /* Name of function */
                n_times      /* Argument(s) */
            );
        }
    };

    PYBIND11_PLUGIN(example) {
        py::module m("example", "pybind11 example plugin");

        py::class_<Animal, std::unique_ptr<Animal>, PyAnimal> animal(m, "Animal");
        animal
            .def(py::init<>())
            .def("go", &Animal::go);

        py::class_<Dog>(m, "Dog", animal)
            .def(py::init<>());

        m.def("call_go", [](Animal *animal) -> std::string {
            /* Release GIL before calling into (potentially long-running) C++ code */
            py::gil_scoped_release release;
            return call_go(animal);
        });

        return m.ptr();
    }

Passing STL data structures
===========================

When including the additional header file :file:`pybind11/stl.h`, conversions
between ``std::vector<>``, ``std::list<>``, ``std::set<>``, and ``std::map<>``
and the Python ``list``, ``set`` and ``dict`` data structures are automatically
enabled. The types ``std::pair<>`` and ``std::tuple<>`` are already supported
out of the box with just the core :file:`pybind11/pybind11.h` header.

.. note::

    Arbitrary nesting of any of these types is supported.

.. seealso::

    The file :file:`example/example2.cpp` contains a complete example that
    demonstrates how to pass STL data types in more detail.

Binding sequence data types, iterators, the slicing protocol, etc.
==================================================================

Please refer to the supplemental example for details.

.. seealso::

    The file :file:`example/example6.cpp` contains a complete example that
    shows how to bind a sequence data type, including length queries
    (``__len__``), iterators (``__iter__``), the slicing protocol and other
    kinds of useful operations.

Return value policies
=====================

Python and C++ use wildly different ways of managing the memory and lifetime of
objects managed by them. This can lead to issues when creating bindings for
functions that return a non-trivial type. Just by looking at the type
information, it is not clear whether Python should take charge of the returned
value and eventually free its resources, or if this is handled on the C++ side.
For this reason, pybind11 provides a several `return value policy` annotations
that can be passed to the :func:`module::def` and :func:`class_::def`
functions. The default policy is :enum:`return_value_policy::automatic`.

.. tabularcolumns:: |p{0.5\textwidth}|p{0.45\textwidth}|

+--------------------------------------------------+----------------------------------------------------------------------------+
| Return value policy                              | Description                                                                |
+==================================================+============================================================================+
| :enum:`return_value_policy::automatic`           | This is the default return value policy, which falls back to the policy    |
|                                                  | :enum:`return_value_policy::take_ownership` when the return value is a     |
|                                                  | pointer. Otherwise, it uses :enum:`return_value::move` or                  |
|                                                  | :enum:`return_value::copy` for rvalue and lvalue references, respectively. |
|                                                  | See below for a description of what all of these different policies do.    |
+--------------------------------------------------+----------------------------------------------------------------------------+
| :enum:`return_value_policy::automatic_reference` | As above, but use policy :enum:`return_value_policy::reference` when the   |
|                                                  | return value is a pointer. You probably won't need to use this.            |
+--------------------------------------------------+----------------------------------------------------------------------------+
| :enum:`return_value_policy::take_ownership`      | Reference an existing object (i.e. do not create a new copy) and take      |
|                                                  | ownership. Python will call the destructor and delete operator when the    |
|                                                  | object's reference count reaches zero. Undefined behavior ensues when the  |
|                                                  | C++ side does the same..                                                   |
+--------------------------------------------------+----------------------------------------------------------------------------+
| :enum:`return_value_policy::copy`                | Create a new copy of the returned object, which will be owned by Python.   |
|                                                  | This policy is comparably safe because the lifetimes of the two instances  |
|                                                  | are decoupled.                                                             |
+--------------------------------------------------+----------------------------------------------------------------------------+
| :enum:`return_value_policy::move`                | Use ``std::move`` to move the return value contents into a new instance    |
|                                                  | that will be owned by Python. This policy is comparably safe because the   |
|                                                  | lifetimes of the two instances (move source and destination) are decoupled.|
+--------------------------------------------------+----------------------------------------------------------------------------+
| :enum:`return_value_policy::reference`           | Reference an existing object, but do not take ownership. The C++ side is   |
|                                                  | responsible for managing the object's lifetime and deallocating it when    |
|                                                  | it is no longer used. Warning: undefined behavior will ensue when the C++  |
|                                                  | side deletes an object that is still referenced and used by Python.        |
+--------------------------------------------------+----------------------------------------------------------------------------+
| :enum:`return_value_policy::reference_internal`  | This policy only applies to methods and properties. It references the      |
|                                                  | object without taking ownership similar to the above                       |
|                                                  | :enum:`return_value_policy::reference` policy. In contrast to that policy, |
|                                                  | the function or property's implicit ``this`` argument (called the *parent*)|
|                                                  | is considered to be the the owner of the return value (the *child*).       |
|                                                  | pybind11 then couples the lifetime of the parent to the child via a        |
|                                                  | reference relationship that ensures that the parent cannot be garbage      |
|                                                  | collected while Python is still using the child. More advanced variations  |
|                                                  | of this scheme are also possible using combinations of                     |
|                                                  | :enum:`return_value_policy::reference` and the :class:`keep_alive` call    |
|                                                  | policy described next.                                                     |
+--------------------------------------------------+----------------------------------------------------------------------------+

The following example snippet shows a use case of the
:enum:`return_value_policy::reference_internal` policy.

.. code-block:: cpp

    class Example {
    public:
        Internal &get_internal() { return internal; }
    private:
        Internal internal;
    };

    PYBIND11_PLUGIN(example) {
        py::module m("example", "pybind11 example plugin");

        py::class_<Example>(m, "Example")
            .def(py::init<>())
            .def("get_internal", &Example::get_internal, "Return the internal data",
                                 py::return_value_policy::reference_internal);

        return m.ptr();
    }

.. warning::

    Code with invalid call policies might access unitialized memory or free
    data structures multiple times, which can lead to hard-to-debug
    non-determinism and segmentation faults, hence it is worth spending the
    time to understand all the different options in the table above.

.. note::

    The next section on :ref:`call_policies` discusses *call policies* that can be
    specified *in addition* to a return value policy from the list above. Call
    policies indicate reference relationships that can involve both return values
    and parameters of functions.

.. note::

   As an alternative to elaborate call policies and lifetime management logic,
   consider using smart pointers (see the section on :ref:`smart_pointers` for
   details). Smart pointers can tell whether an object is still referenced from
   C++ or Python, which generally eliminates the kinds of inconsistencies that
   can lead to crashes or undefined behavior. For functions returning smart
   pointers, it is not necessary to specify a return value policy.

.. _call_policies:

Additional call policies
========================

In addition to the above return value policies, further `call policies` can be
specified to indicate dependencies between parameters. There is currently just
one policy named ``keep_alive<Nurse, Patient>``, which indicates that the
argument with index ``Patient`` should be kept alive at least until the
argument with index ``Nurse`` is freed by the garbage collector; argument
indices start at one, while zero refers to the return value. For methods, index
one refers to the implicit ``this`` pointer, while regular arguments begin at
index two. Arbitrarily many call policies can be specified.

Consider the following example: the binding code for a list append operation
that ties the lifetime of the newly added element to the underlying container
might be declared as follows:

.. code-block:: cpp

    py::class_<List>(m, "List")
        .def("append", &List::append, py::keep_alive<1, 2>());

.. note::

    ``keep_alive`` is analogous to the ``with_custodian_and_ward`` (if Nurse,
    Patient != 0) and ``with_custodian_and_ward_postcall`` (if Nurse/Patient ==
    0) policies from Boost.Python.

.. seealso::

    The file :file:`example/example13.cpp` contains a complete example that
    demonstrates using :class:`keep_alive` in more detail.

Implicit type conversions
=========================

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

Unique pointers
===============

Given a class ``Example`` with Python bindings, it's possible to return
instances wrapped in C++11 unique pointers, like so

.. code-block:: cpp

    std::unique_ptr<Example> create_example() { return std::unique_ptr<Example>(new Example()); }

.. code-block:: cpp

    m.def("create_example", &create_example);

In other words, there is nothing special that needs to be done. While returning
unique pointers in this way is allowed, it is *illegal* to use them as function
arguments. For instance, the following function signature cannot be processed
by pybind11.

.. code-block:: cpp

    void do_something_with_example(std::unique_ptr<Example> ex) { ... }

The above signature would imply that Python needs to give up ownership of an
object that is passed to this function, which is generally not possible (for
instance, the object might be referenced elsewhere).

.. _smart_pointers:

Smart pointers
==============

This section explains how to pass values that are wrapped in "smart" pointer
types with internal reference counting. For the simpler C++11 unique pointers,
refer to the previous section.

The binding generator for classes, :class:`class_`, takes an optional second
template type, which denotes a special *holder* type that is used to manage
references to the object. When wrapping a type named ``Type``, the default
value of this template parameter is ``std::unique_ptr<Type>``, which means that
the object is deallocated when Python's reference count goes to zero.

It is possible to switch to other types of reference counting wrappers or smart
pointers, which is useful in codebases that rely on them. For instance, the
following snippet causes ``std::shared_ptr`` to be used instead.

.. code-block:: cpp

    py::class_<Example, std::shared_ptr<Example> /* <- holder type */> obj(m, "Example");

Note that any particular class can only be associated with a single holder type.

To enable transparent conversions for functions that take shared pointers as an
argument or that return them, a macro invocation similar to the following must
be declared at the top level before any binding code:

.. code-block:: cpp

    PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

.. note::

    The first argument of :func:`PYBIND11_DECLARE_HOLDER_TYPE` should be a
    placeholder name that is used as a template parameter of the second
    argument. Thus, feel free to use any identifier, but use it consistently on
    both sides; also, don't use the name of a type that already exists in your
    codebase.

One potential stumbling block when using holder types is that they need to be
applied consistently. Can you guess what's broken about the following binding
code?

.. code-block:: cpp

    class Child { };

    class Parent {
    public:
       Parent() : child(std::make_shared<Child>()) { }
       Child *get_child() { return child.get(); }  /* Hint: ** DON'T DO THIS ** */
    private:
        std::shared_ptr<Child> child;
    };

    PYBIND11_PLUGIN(example) {
        py::module m("example");

        py::class_<Child, std::shared_ptr<Child>>(m, "Child");

        py::class_<Parent, std::shared_ptr<Parent>>(m, "Parent")
           .def(py::init<>())
           .def("get_child", &Parent::get_child);

        return m.ptr();
    }

The following Python code will cause undefined behavior (and likely a
segmentation fault).

.. code-block:: python

   from example import Parent
   print(Parent().get_child())

The problem is that ``Parent::get_child()`` returns a pointer to an instance of
``Child``, but the fact that this instance is already managed by
``std::shared_ptr<...>`` is lost when passing raw pointers. In this case,
pybind11 will create a second independent ``std::shared_ptr<...>`` that also
claims ownership of the pointer. In the end, the object will be freed **twice**
since these shared pointers have no way of knowing about each other.

There are two ways to resolve this issue:

1. For types that are managed by a smart pointer class, never use raw pointers
   in function arguments or return values. In other words: always consistently
   wrap pointers into their designated holder types (such as
   ``std::shared_ptr<...>``). In this case, the signature of ``get_child()``
   should be modified as follows:

.. code-block:: cpp

    std::shared_ptr<Child> get_child() { return child; }

2. Adjust the definition of ``Child`` by specifying
   ``std::enable_shared_from_this<T>`` (see cppreference_ for details) as a
   base class. This adds a small bit of information to ``Child`` that allows
   pybind11 to realize that there is already an existing
   ``std::shared_ptr<...>`` and communicate with it. In this case, the
   declaration of ``Child`` should look as follows:

.. _cppreference: http://en.cppreference.com/w/cpp/memory/enable_shared_from_this

.. code-block:: cpp

    class Child : public std::enable_shared_from_this<Child> { };

.. seealso::

    The file :file:`example/example8.cpp` contains a complete example that
    demonstrates how to work with custom reference-counting holder types in
    more detail.

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

Catching and throwing exceptions
================================

When C++ code invoked from Python throws an ``std::exception``, it is
automatically converted into a Python ``Exception``. pybind11 defines multiple
special exception classes that will map to different types of Python
exceptions:

.. tabularcolumns:: |p{0.5\textwidth}|p{0.45\textwidth}|

+--------------------------------------+------------------------------+
|  C++ exception type                  |  Python exception type       |
+======================================+==============================+
| :class:`std::exception`              | ``RuntimeError``             |
+--------------------------------------+------------------------------+
| :class:`std::bad_alloc`              | ``MemoryError``              |
+--------------------------------------+------------------------------+
| :class:`std::domain_error`           | ``ValueError``               |
+--------------------------------------+------------------------------+
| :class:`std::invalid_argument`       | ``ValueError``               |
+--------------------------------------+------------------------------+
| :class:`std::length_error`           | ``ValueError``               |
+--------------------------------------+------------------------------+
| :class:`std::out_of_range`           | ``ValueError``               |
+--------------------------------------+------------------------------+
| :class:`std::range_error`            | ``ValueError``               |
+--------------------------------------+------------------------------+
| :class:`pybind11::stop_iteration`    | ``StopIteration`` (used to   |
|                                      | implement custom iterators)  |
+--------------------------------------+------------------------------+
| :class:`pybind11::index_error`       | ``IndexError`` (used to      |
|                                      | indicate out of bounds       |
|                                      | accesses in ``__getitem__``, |
|                                      | ``__setitem__``, etc.)       |
+--------------------------------------+------------------------------+
| :class:`pybind11::value_error`       | ``ValueError`` (used to      |
|                                      | indicate wrong value passed  |
|                                      | in ``container.remove(...)`` |
+--------------------------------------+------------------------------+
| :class:`pybind11::error_already_set` | Indicates that the Python    |
|                                      | exception flag has already   |
|                                      | been initialized             |
+--------------------------------------+------------------------------+

When a Python function invoked from C++ throws an exception, it is converted
into a C++ exception of type :class:`error_already_set` whose string payload
contains a textual summary.

There is also a special exception :class:`cast_error` that is thrown by
:func:`handle::call` when the input arguments cannot be converted to Python
objects.

.. _opaque:

Treating STL data structures as opaque objects
==============================================

pybind11 heavily relies on a template matching mechanism to convert parameters
and return values that are constructed from STL data types such as vectors,
linked lists, hash tables, etc. This even works in a recursive manner, for
instance to deal with lists of hash maps of pairs of elementary and custom
types, etc.

However, a fundamental limitation of this approach is that internal conversions
between Python and C++ types involve a copy operation that prevents
pass-by-reference semantics. What does this mean?

Suppose we bind the following function

.. code-block:: cpp

    void append_1(std::vector<int> &v) {
       v.push_back(1);
    }

and call it from Python, the following happens:

.. code-block:: python

   >>> v = [5, 6]
   >>> append_1(v)
   >>> print(v)
   [5, 6]

As you can see, when passing STL data structures by reference, modifications
are not propagated back the Python side. A similar situation arises when
exposing STL data structures using the ``def_readwrite`` or ``def_readonly``
functions:

.. code-block:: cpp

    /* ... definition ... */

    class MyClass {
        std::vector<int> contents;
    };

    /* ... binding code ... */

    py::class_<MyClass>(m, "MyClass")
        .def(py::init<>)
        .def_readwrite("contents", &MyClass::contents);

In this case, properties can be read and written in their entirety. However, an
``append`` operaton involving such a list type has no effect:

.. code-block:: python

   >>> m = MyClass()
   >>> m.contents = [5, 6]
   >>> print(m.contents)
   [5, 6]
   >>> m.contents.append(7)
   >>> print(m.contents)
   [5, 6]

To deal with both of the above situations, pybind11 provides a macro named
``PYBIND11_MAKE_OPAQUE(T)`` that disables the template-based conversion
machinery of types, thus rendering them *opaque*. The contents of opaque
objects are never inspected or extracted, hence they can be passed by
reference. For instance, to turn ``std::vector<int>`` into an opaque type, add
the declaration

.. code-block:: cpp

    PYBIND11_MAKE_OPAQUE(std::vector<int>);

before any binding code (e.g. invocations to ``class_::def()``, etc.). This
macro must be specified at the top level, since instantiates a partial template
overload. If your binding code consists of multiple compilation units, it must
be present in every file preceding any usage of ``std::vector<int>``. Opaque
types must also have a corresponding ``class_`` declaration to associate them
with a name in Python, and to define a set of available operations:

.. code-block:: cpp

    py::class_<std::vector<int>>(m, "IntVector")
        .def(py::init<>())
        .def("clear", &std::vector<int>::clear)
        .def("pop_back", &std::vector<int>::pop_back)
        .def("__len__", [](const std::vector<int> &v) { return v.size(); })
        .def("__iter__", [](std::vector<int> &v) {
           return py::make_iterator(v.begin(), v.end());
        }, py::keep_alive<0, 1>()) /* Keep vector alive while iterator is used */
        // ....


.. seealso::

    The file :file:`example/example14.cpp` contains a complete example that
    demonstrates how to create and expose opaque types using pybind11 in more
    detail.

.. _eigen:

Transparent conversion of dense and sparse Eigen data types
===========================================================

Eigen [#f1]_ is C++ header-based library for dense and sparse linear algebra. Due to
its popularity and widespread adoption, pybind11 provides transparent
conversion support between Eigen and Scientific Python linear algebra data types.

Specifically, when including the optional header file :file:`pybind11/eigen.h`,
pybind11 will automatically and transparently convert

1. Static and dynamic Eigen dense vectors and matrices to instances of
   ``numpy.ndarray`` (and vice versa).

1. Eigen sparse vectors and matrices to instances of
   ``scipy.sparse.csr_matrix``/``scipy.sparse.csc_matrix`` (and vice versa).

This makes it possible to bind most kinds of functions that rely on these types.
One major caveat are functions that take Eigen matrices *by reference* and modify
them somehow, in which case the information won't be propagated to the caller.

.. code-block:: cpp

    /* The Python bindings of this function won't replicate
       the intended effect of modifying the function argument */
    void scale_by_2(Eigen::Vector3f &v) {
       v *= 2;
    }

To see why this is, refer to the section on :ref:`opaque` (although that
section specifically covers STL data types, the underlying issue is the same).
The next two sections discuss an efficient alternative for exposing the
underlying native Eigen types as opaque objects in a way that still integrates
with NumPy and SciPy.

.. [#f1] http://eigen.tuxfamily.org

.. seealso::

    The file :file:`example/eigen.cpp` contains a complete example that
    shows how to pass Eigen sparse and dense data types in more detail.

Buffer protocol
===============

Python supports an extremely general and convenient approach for exchanging
data between plugin libraries. Types can expose a buffer view [#f2]_, which
provides fast direct access to the raw internal data representation. Suppose we
want to bind the following simplistic Matrix class:

.. code-block:: cpp

    class Matrix {
    public:
        Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols) {
            m_data = new float[rows*cols];
        }
        float *data() { return m_data; }
        size_t rows() const { return m_rows; }
        size_t cols() const { return m_cols; }
    private:
        size_t m_rows, m_cols;
        float *m_data;
    };

The following binding code exposes the ``Matrix`` contents as a buffer object,
making it possible to cast Matrices into NumPy arrays. It is even possible to
completely avoid copy operations with Python expressions like
``np.array(matrix_instance, copy = False)``.

.. code-block:: cpp

    py::class_<Matrix>(m, "Matrix")
       .def_buffer([](Matrix &m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),                            /* Pointer to buffer */
                sizeof(float),                       /* Size of one scalar */
                py::format_descriptor<float>::value, /* Python struct-style format descriptor */
                2,                                   /* Number of dimensions */
                { m.rows(), m.cols() },              /* Buffer dimensions */
                { sizeof(float) * m.rows(),          /* Strides (in bytes) for each index */
                  sizeof(float) }
            );
        });

The snippet above binds a lambda function, which can create ``py::buffer_info``
description records on demand describing a given matrix. The contents of
``py::buffer_info`` mirror the Python buffer protocol specification.

.. code-block:: cpp

    struct buffer_info {
        void *ptr;
        size_t itemsize;
        std::string format;
        int ndim;
        std::vector<size_t> shape;
        std::vector<size_t> strides;
    };

To create a C++ function that can take a Python buffer object as an argument,
simply use the type ``py::buffer`` as one of its arguments. Buffers can exist
in a great variety of configurations, hence some safety checks are usually
necessary in the function body. Below, you can see an basic example on how to
define a custom constructor for the Eigen double precision matrix
(``Eigen::MatrixXd``) type, which supports initialization from compatible
buffer objects (e.g. a NumPy matrix).

.. code-block:: cpp

    /* Bind MatrixXd (or some other Eigen type) to Python */
    typedef Eigen::MatrixXd Matrix;

    typedef Matrix::Scalar Scalar;
    constexpr bool rowMajor = Matrix::Flags & Eigen::RowMajorBit;

    py::class_<Matrix>(m, "Matrix")
        .def("__init__", [](Matrix &m, py::buffer b) {
            typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> Strides;

            /* Request a buffer descriptor from Python */
            py::buffer_info info = b.request();

            /* Some sanity checks ... */
            if (info.format != py::format_descriptor<Scalar>::value)
                throw std::runtime_error("Incompatible format: expected a double array!");

            if (info.ndim != 2)
                throw std::runtime_error("Incompatible buffer dimension!");

            auto strides = Strides(
                info.strides[rowMajor ? 0 : 1] / sizeof(Scalar),
                info.strides[rowMajor ? 1 : 0] / sizeof(Scalar));

            auto map = Eigen::Map<Matrix, 0, Strides>(
                static_cat<Scalar *>(info.ptr), info.shape[0], info.shape[1], strides);

            new (&m) Matrix(map);
        });

For reference, the ``def_buffer()`` call for this Eigen data type should look
as follows:

.. code-block:: cpp

    .def_buffer([](Matrix &m) -> py::buffer_info {
        return py::buffer_info(
            m.data(),                /* Pointer to buffer */
            sizeof(Scalar),          /* Size of one scalar */
            /* Python struct-style format descriptor */
            py::format_descriptor<Scalar>::value,
            /* Number of dimensions */
            2,
            /* Buffer dimensions */
            { (size_t) m.rows(),
              (size_t) m.cols() },
            /* Strides (in bytes) for each index */
            { sizeof(Scalar) * (rowMajor ? m.cols() : 1),
              sizeof(Scalar) * (rowMajor ? 1 : m.rows()) }
        );
     })

For a much easier approach of binding Eigen types (although with some
limitations), refer to the section on :ref:`eigen`.

.. seealso::

    The file :file:`example/example7.cpp` contains a complete example that
    demonstrates using the buffer protocol with pybind11 in more detail.

.. [#f2] http://docs.python.org/3/c-api/buffer.html

NumPy support
=============

By exchanging ``py::buffer`` with ``py::array`` in the above snippet, we can
restrict the function so that it only accepts NumPy arrays (rather than any
type of Python object satisfying the buffer protocol).

In many situations, we want to define a function which only accepts a NumPy
array of a certain data type. This is possible via the ``py::array_t<T>``
template. For instance, the following function requires the argument to be a
NumPy array containing double precision values.

.. code-block:: cpp

    void f(py::array_t<double> array);

When it is invoked with a different type (e.g. an integer or a list of
integers), the binding code will attempt to cast the input into a NumPy array
of the requested type. Note that this feature requires the
:file:``pybind11/numpy.h`` header to be included.

Data in NumPy arrays is not guaranteed to packed in a dense manner;
furthermore, entries can be separated by arbitrary column and row strides.
Sometimes, it can be useful to require a function to only accept dense arrays
using either the C (row-major) or Fortran (column-major) ordering. This can be
accomplished via a second template argument with values ``py::array::c_style``
or ``py::array::f_style``.

.. code-block:: cpp

    void f(py::array_t<double, py::array::c_style | py::array::forcecast> array);

The ``py::array::forcecast`` argument is the default value of the second
template paramenter, and it ensures that non-conforming arguments are converted
into an array satisfying the specified requirements instead of trying the next
function overload.

Vectorizing functions
=====================

Suppose we want to bind a function with the following signature to Python so
that it can process arbitrary NumPy array arguments (vectors, matrices, general
N-D arrays) in addition to its normal arguments:

.. code-block:: cpp

    double my_func(int x, float y, double z);

After including the ``pybind11/numpy.h`` header, this is extremely simple:

.. code-block:: cpp

    m.def("vectorized_func", py::vectorize(my_func));

Invoking the function like below causes 4 calls to be made to ``my_func`` with
each of the array elements. The significant advantage of this compared to
solutions like ``numpy.vectorize()`` is that the loop over the elements runs
entirely on the C++ side and can be crunched down into a tight, optimized loop
by the compiler. The result is returned as a NumPy array of type
``numpy.dtype.float64``.

.. code-block:: python

    >>> x = np.array([[1, 3],[5, 7]])
    >>> y = np.array([[2, 4],[6, 8]])
    >>> z = 3
    >>> result = vectorized_func(x, y, z)

The scalar argument ``z`` is transparently replicated 4 times.  The input
arrays ``x`` and ``y`` are automatically converted into the right types (they
are of type  ``numpy.dtype.int64`` but need to be ``numpy.dtype.int32`` and
``numpy.dtype.float32``, respectively)

Sometimes we might want to explicitly exclude an argument from the vectorization
because it makes little sense to wrap it in a NumPy array. For instance,
suppose the function signature was

.. code-block:: cpp

    double my_func(int x, float y, my_custom_type *z);

This can be done with a stateful Lambda closure:

.. code-block:: cpp

    // Vectorize a lambda function with a capture object (e.g. to exclude some arguments from the vectorization)
    m.def("vectorized_func",
        [](py::array_t<int> x, py::array_t<float> y, my_custom_type *z) {
            auto stateful_closure = [z](int x, float y) { return my_func(x, y, z); };
            return py::vectorize(stateful_closure)(x, y);
        }
    );

In cases where the computation is too complicated to be reduced to
``vectorize``, it will be necessary to create and access the buffer contents
manually. The following snippet contains a complete example that shows how this
works (the code is somewhat contrived, since it could have been done more
simply using ``vectorize``).

.. code-block:: cpp

    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>

    namespace py = pybind11;

    py::array_t<double> add_arrays(py::array_t<double> input1, py::array_t<double> input2) {
        auto buf1 = input1.request(), buf2 = input2.request();

        if (buf1.ndim != 1 || buf2.ndim != 1)
            throw std::runtime_error("Number of dimensions must be one");

        if (buf1.shape[0] != buf2.shape[0])
            throw std::runtime_error("Input shapes must match");

        auto result = py::array(py::buffer_info(
            nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
            sizeof(double),     /* Size of one item */
            py::format_descriptor<double>::value(), /* Buffer format */
            buf1.ndim,          /* How many dimensions? */
            { buf1.shape[0] },  /* Number of elements for each dimension */
            { sizeof(double) }  /* Strides for each dimension */
        ));

        auto buf3 = result.request();

        double *ptr1 = (double *) buf1.ptr,
               *ptr2 = (double *) buf2.ptr,
               *ptr3 = (double *) buf3.ptr;

        for (size_t idx = 0; idx < buf1.shape[0]; idx++)
            ptr3[idx] = ptr1[idx] + ptr2[idx];

        return result;
    }

    PYBIND11_PLUGIN(test) {
        py::module m("test");
        m.def("add_arrays", &add_arrays, "Add two NumPy arrays");
        return m.ptr();
    }

.. seealso::

    The file :file:`example/example10.cpp` contains a complete example that
    demonstrates using :func:`vectorize` in more detail.

Functions taking Python objects as arguments
============================================

pybind11 exposes all major Python types using thin C++ wrapper classes. These
wrapper classes can also be used as parameters of functions in bindings, which
makes it possible to directly work with native Python types on the C++ side.
For instance, the following statement iterates over a Python ``dict``:

.. code-block:: cpp

    void print_dict(py::dict dict) {
        /* Easily interact with Python types */
        for (auto item : dict)
            std::cout << "key=" << item.first << ", "
                      << "value=" << item.second << std::endl;
    }

Available types include :class:`handle`, :class:`object`, :class:`bool_`,
:class:`int_`, :class:`float_`, :class:`str`, :class:`bytes`, :class:`tuple`,
:class:`list`, :class:`dict`, :class:`slice`, :class:`none`, :class:`capsule`,
:class:`iterable`, :class:`iterator`, :class:`function`, :class:`buffer`,
:class:`array`, and :class:`array_t`.

In this kind of mixed code, it is often necessary to convert arbitrary C++
types to Python, which can be done using :func:`cast`:

.. code-block:: cpp

    MyClass *cls = ..;
    py::object obj = py::cast(cls);

The reverse direction uses the following syntax:

.. code-block:: cpp

    py::object obj = ...;
    MyClass *cls = obj.cast<MyClass *>();

When conversion fails, both directions throw the exception :class:`cast_error`.
It is also possible to call python functions via ``operator()``.

.. code-block:: cpp

    py::function f = <...>;
    py::object result_py = f(1234, "hello", some_instance);
    MyClass &result = result_py.cast<MyClass>();

The special ``f(*args)`` and ``f(*args, **kwargs)`` syntax is also supported to
supply arbitrary argument and keyword lists, although these cannot be mixed
with other parameters.

.. code-block:: cpp

    py::function f = <...>;
    py::tuple args = py::make_tuple(1234);
    py::dict kwargs;
    kwargs["y"] = py::cast(5678);
    py::object result = f(*args, **kwargs);

.. seealso::

    The file :file:`example/example2.cpp` contains a complete example that
    demonstrates passing native Python types in more detail. The file
    :file:`example/example11.cpp` discusses usage of ``args`` and ``kwargs``.

Default arguments revisited
===========================

The section on :ref:`default_args` previously discussed basic usage of default
arguments using pybind11. One noteworthy aspect of their implementation is that
default arguments are converted to Python objects right at declaration time.
Consider the following example:

.. code-block:: cpp

    py::class_<MyClass>("MyClass")
        .def("myFunction", py::arg("arg") = SomeType(123));

In this case, pybind11 must already be set up to deal with values of the type
``SomeType`` (via a prior instantiation of ``py::class_<SomeType>``), or an
exception will be thrown.

Another aspect worth highlighting is that the "preview" of the default argument
in the function signature is generated using the object's ``__repr__`` method.
If not available, the signature may not be very helpful, e.g.:

.. code-block:: python

    FUNCTIONS
    ...
    |  myFunction(...)
    |      Signature : (MyClass, arg : SomeType = <SomeType object at 0x101b7b080>) -> NoneType
    ...

The first way of addressing this is by defining ``SomeType.__repr__``.
Alternatively, it is possible to specify the human-readable preview of the
default argument manually using the ``arg_t`` notation:

.. code-block:: cpp

    py::class_<MyClass>("MyClass")
        .def("myFunction", py::arg_t<SomeType>("arg", SomeType(123), "SomeType(123)"));

Sometimes it may be necessary to pass a null pointer value as a default
argument. In this case, remember to cast it to the underlying type in question,
like so:

.. code-block:: cpp

    py::class_<MyClass>("MyClass")
        .def("myFunction", py::arg("arg") = (SomeType *) nullptr);

Binding functions that accept arbitrary numbers of arguments and keywords arguments
===================================================================================

Python provides a useful mechanism to define functions that accept arbitrary
numbers of arguments and keyword arguments:

.. code-block:: cpp

   def generic(*args, **kwargs):
       # .. do something with args and kwargs

Such functions can also be created using pybind11:

.. code-block:: cpp

   void generic(py::args args, py::kwargs kwargs) {
       /// .. do something with args
       if (kwargs)
           /// .. do something with kwargs
   }

   /// Binding code
   m.def("generic", &generic);

(See ``example/example11.cpp``). The class ``py::args`` derives from
``py::list`` and ``py::kwargs`` derives from ``py::dict`` Note that the
``kwargs`` argument is invalid if no keyword arguments were actually provided.
Please refer to the other examples for details on how to iterate over these,
and on how to cast their entries into C++ objects.

Partitioning code over multiple extension modules
=================================================

It's straightforward to split binding code over multiple extension modules,
while referencing types that are declared elsewhere. Everything "just" works
without any special precautions. One exception to this rule occurs when
extending a type declared in another extension module. Recall the basic example
from Section :ref:`inheritance`.

.. code-block:: cpp

    py::class_<Pet> pet(m, "Pet");
    pet.def(py::init<const std::string &>())
       .def_readwrite("name", &Pet::name);

    py::class_<Dog>(m, "Dog", pet /* <- specify parent */)
        .def(py::init<const std::string &>())
        .def("bark", &Dog::bark);

Suppose now that ``Pet`` bindings are defined in a module named ``basic``,
whereas the ``Dog`` bindings are defined somewhere else. The challenge is of
course that the variable ``pet`` is not available anymore though it is needed
to indicate the inheritance relationship to the constructor of ``class_<Dog>``.
However, it can be acquired as follows:

.. code-block:: cpp

    py::object pet = (py::object) py::module::import("basic").attr("Pet");

    py::class_<Dog>(m, "Dog", pet)
        .def(py::init<const std::string &>())
        .def("bark", &Dog::bark);

Alternatively, we can rely on the ``base`` tag, which performs an automated
lookup of the corresponding Python type. However, this also requires invoking
the ``import`` function once to ensure that the pybind11 binding code of the
module ``basic`` has been executed.

.. code-block:: cpp

    py::module::import("basic");

    py::class_<Dog>(m, "Dog", py::base<Pet>())
        .def(py::init<const std::string &>())
        .def("bark", &Dog::bark);

Naturally, both methods will fail when there are cyclic dependencies.

Note that compiling code which has its default symbol visibility set to
*hidden* (e.g. via the command line flag ``-fvisibility=hidden`` on GCC/Clang) can interfere with the
ability to access types defined in another extension module. Workarounds
include changing the global symbol visibility (not recommended, because it will
lead unnecessarily large binaries) or manually exporting types that are
accessed by multiple extension modules:

.. code-block:: cpp

    #ifdef _WIN32
    #  define EXPORT_TYPE __declspec(dllexport)
    #else
    #  define EXPORT_TYPE __attribute__ ((visibility("default")))
    #endif

    class EXPORT_TYPE Dog : public Animal {
        ...
    };


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

    The file :file:`example/example15.cpp` contains a complete example that
    demonstrates how to pickle and unpickle types using pybind11 in more detail.

.. [#f3] http://docs.python.org/3/library/pickle.html#pickling-class-instances

Generating documentation using Sphinx
=====================================

Sphinx [#f4]_ has the ability to inspect the signatures and documentation
strings in pybind11-based extension modules to automatically generate beautiful
documentation in a variety formats. The pbtest repository [#f5]_ contains a
simple example repository which uses this approach.

There are two potential gotchas when using this approach: first, make sure that
the resulting strings do not contain any :kbd:`TAB` characters, which break the
docstring parsing routines. You may want to use C++11 raw string literals,
which are convenient for multi-line comments. Conveniently, any excess
indentation will be automatically be removed by Sphinx. However, for this to
work, it is important that all lines are indented consistently, i.e.:

.. code-block:: cpp

    // ok
    m.def("foo", &foo, R"mydelimiter(
        The foo function

        Parameters
        ----------
    )mydelimiter");

    // *not ok*
    m.def("foo", &foo, R"mydelimiter(The foo function

        Parameters
        ----------
    )mydelimiter");

.. [#f4] http://www.sphinx-doc.org
.. [#f5] http://github.com/pybind/pbtest
