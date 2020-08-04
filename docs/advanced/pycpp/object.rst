Python types
############

Available wrappers
==================

All major Python types are available as thin C++ wrapper classes. These
can also be used as function parameters -- see :ref:`python_objects_as_args`.

Available types include :class:`handle`, :class:`object`, :class:`bool_`,
:class:`int_`, :class:`float_`, :class:`str`, :class:`bytes`, :class:`tuple`,
:class:`list`, :class:`dict`, :class:`slice`, :class:`none`, :class:`capsule`,
:class:`iterable`, :class:`iterator`, :class:`function`, :class:`buffer`,
:class:`array`, and :class:`array_t`.

.. warning::

    You should be aware of how these classes interact with :func:`py::none`.
    See :ref:`pytypes_interaction_with_none` for more details.

Casting back and forth
======================

In this kind of mixed code, it is often necessary to convert arbitrary C++
types to Python, which can be done using :func:`py::cast`:

.. code-block:: cpp

    MyClass *cls = ..;
    py::object obj = py::cast(cls);

The reverse direction uses the following syntax:

.. code-block:: cpp

    py::object obj = ...;
    MyClass *cls = obj.cast<MyClass *>();

When conversion fails, both directions throw the exception :class:`cast_error`.

.. _python_libs:

Accessing Python libraries from C++
===================================

It is also possible to import objects defined in the Python standard
library or available in the current Python environment (``sys.path``) and work
with these in C++.

This example obtains a reference to the Python ``Decimal`` class.

.. code-block:: cpp

    // Equivalent to "from decimal import Decimal"
    py::object Decimal = py::module::import("decimal").attr("Decimal");

.. code-block:: cpp

    // Try to import scipy
    py::object scipy = py::module::import("scipy");
    return scipy.attr("__version__");

.. _calling_python_functions:

Calling Python functions
========================

It is also possible to call Python classes, functions and methods
via ``operator()``.

.. code-block:: cpp

    // Construct a Python object of class Decimal
    py::object pi = Decimal("3.14159");

.. code-block:: cpp

    // Use Python to make our directories
    py::object os = py::module::import("os");
    py::object makedirs = os.attr("makedirs");
    makedirs("/tmp/path/to/somewhere");

One can convert the result obtained from Python to a pure C++ version
if a ``py::class_`` or type conversion is defined.

.. code-block:: cpp

    py::function f = <...>;
    py::object result_py = f(1234, "hello", some_instance);
    MyClass &result = result_py.cast<MyClass>();

.. _calling_python_methods:

Calling Python methods
========================

To call an object's method, one can again use ``.attr`` to obtain access to the
Python method.

.. code-block:: cpp

    // Calculate e^π in decimal
    py::object exp_pi = pi.attr("exp")();
    py::print(py::str(exp_pi));

In the example above ``pi.attr("exp")`` is a *bound method*: it will always call
the method for that same instance of the class. Alternately one can create an
*unbound method* via the Python class (instead of instance) and pass the ``self``
object explicitly, followed by other arguments.

.. code-block:: cpp

    py::object decimal_exp = Decimal.attr("exp");

    // Compute the e^n for n=0..4
    for (int n = 0; n < 5; n++) {
        py::print(decimal_exp(Decimal(n));
    }

Keyword arguments
=================

Keyword arguments are also supported. In Python, there is the usual call syntax:

.. code-block:: python

    def f(number, say, to):
        ...  # function code

    f(1234, say="hello", to=some_instance)  # keyword call in Python

In C++, the same call can be made using:

.. code-block:: cpp

    using namespace pybind11::literals; // to bring in the `_a` literal
    f(1234, "say"_a="hello", "to"_a=some_instance); // keyword call in C++

Unpacking arguments
===================

Unpacking of ``*args`` and ``**kwargs`` is also possible and can be mixed with
other arguments:

.. code-block:: cpp

    // * unpacking
    py::tuple args = py::make_tuple(1234, "hello", some_instance);
    f(*args);

    // ** unpacking
    py::dict kwargs = py::dict("number"_a=1234, "say"_a="hello", "to"_a=some_instance);
    f(**kwargs);

    // mixed keywords, * and ** unpacking
    py::tuple args = py::make_tuple(1234);
    py::dict kwargs = py::dict("to"_a=some_instance);
    f(*args, "say"_a="hello", **kwargs);

Generalized unpacking according to PEP448_ is also supported:

.. code-block:: cpp

    py::dict kwargs1 = py::dict("number"_a=1234);
    py::dict kwargs2 = py::dict("to"_a=some_instance);
    f(**kwargs1, "say"_a="hello", **kwargs2);

.. seealso::

    The file :file:`tests/test_pytypes.cpp` contains a complete
    example that demonstrates passing native Python types in more detail. The
    file :file:`tests/test_callbacks.cpp` presents a few examples of calling
    Python functions from C++, including keywords arguments and unpacking.

.. _PEP448: https://www.python.org/dev/peps/pep-0448/

.. _pytypes_interaction_with_none:

Interaction with None
=====================

You may be tempted to use types like ``py::str`` and ``py::dict`` in C++
signatures (either pure C++, or in bound signatures). However, there are some
"gotchas" for ``py::none()`` and how it interacts with these types. In best
case scenarios, it will fail fast (e.g. with default arguments); in worst
cases, it will silently work but corrupt the types you want to work with.

In general, the pytypes like ``py::str``, ``py::dict``, etc., are
strict **non-nullable** reference types. They may not store a copy when
assigned to, but they cannot store ``None``. For statically typed languages,
this is in contrast  Java's ``String`` or ``List<E>``, or C#'s ``string`` or
``List<T>``, which are strict nullable refernce types, and C++'s
``std::string``, which is simply a value type, or
``std::optional<std::string>``, which is a nullable value type.

At a first glance, you may think after executing the following code, the
expression ``my_value.is(py::none())`` will be true:

.. code-block:: cpp

    py::str my_value = py::none();

However, this is not the case. Instead, the value of ``my_value`` will be equal
to the Python value of ``str(None)``, due to how :c:macro:`PYBIND11_OBJECT_CVT`
is used in :file:`pybind11/pytypes.h`.

Additionally, calling the following binding with the default argument used will
raise a ``TypeError`` about invalid arguments:

.. code-block:: cpp

    m.def(
        "my_function",
        [](py::str my_value) { ... },
        py::arg("my_value") = py::none());

In both of these cases where you may want to pass ``None`` through any
signatures where you want to constrain the type, you should either use
:class:`py::object` in conjunction with :func:`py::isinstance`, or use the
corresponding C++ type with `std::optional` (if it is available on your
system).

An example of working around the above edge case for conversion:

.. code-block:: cpp

    py::object my_value = /* py::none() or some string */;
    ...
    if (!my_value.is(py::none()) && !py::isinstance<py::str>(my_value)) {
        /* error behavior */
    }

An example of working around the above edge case for default arguments:

.. code-block:: cpp

    m.def(
        "my_function",
        [](std::optional<std::string> my_value) { ... },
        py::arg("my_value") = std::nullopt);

For more details, see the tests for ``pytypes`` mentioned above.
