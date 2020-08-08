Exceptions
##########

Built-in exception translation
==============================

When C++ code invoked from Python throws an ``std::exception``, it is
automatically converted into a Python ``Exception``. pybind11 defines multiple
special exception classes that will map to different types of Python
exceptions:

.. tabularcolumns:: |p{0.5\textwidth}|p{0.45\textwidth}|

+--------------------------------------+--------------------------------------+
|  C++ exception type                  |  Python exception type               |
+======================================+======================================+
| :class:`std::exception`              | ``RuntimeError``                     |
+--------------------------------------+--------------------------------------+
| :class:`std::bad_alloc`              | ``MemoryError``                      |
+--------------------------------------+--------------------------------------+
| :class:`std::domain_error`           | ``ValueError``                       |
+--------------------------------------+--------------------------------------+
| :class:`std::invalid_argument`       | ``ValueError``                       |
+--------------------------------------+--------------------------------------+
| :class:`std::length_error`           | ``ValueError``                       |
+--------------------------------------+--------------------------------------+
| :class:`std::out_of_range`           | ``IndexError``                       |
+--------------------------------------+--------------------------------------+
| :class:`std::range_error`            | ``ValueError``                       |
+--------------------------------------+--------------------------------------+
| :class:`std::overflow_error`         | ``OverflowError``                    |
+--------------------------------------+--------------------------------------+
| :class:`pybind11::stop_iteration`    | ``StopIteration`` (used to implement |
|                                      | custom iterators)                    |
+--------------------------------------+--------------------------------------+
| :class:`pybind11::index_error`       | ``IndexError`` (used to indicate out |
|                                      | of bounds access in ``__getitem__``, |
|                                      | ``__setitem__``, etc.)               |
+--------------------------------------+--------------------------------------+
| :class:`pybind11::value_error`       | ``ValueError`` (used to indicate     |
|                                      | wrong value passed in                |
|                                      | ``container.remove(...)``)           |
+--------------------------------------+--------------------------------------+
| :class:`pybind11::key_error`         | ``KeyError`` (used to indicate out   |
|                                      | of bounds access in ``__getitem__``, |
|                                      | ``__setitem__`` in dict-like         |
|                                      | objects, etc.)                       |
+--------------------------------------+--------------------------------------+
| :class:`pybind11::error_already_set` | Indicates that the Python exception  |
|                                      | flag has already been set via Python |
|                                      | API calls from C++ code; this C++    |
|                                      | exception is used to propagate such  |
|                                      | a Python exception back to Python.   |
+--------------------------------------+--------------------------------------+

When a Python function invoked from C++ throws an exception, pybind11 will convert
it into a C++ exception of type :class:`error_already_set` whose string payload
contains a textual summary. If you call the Python C-API directly, and it
returns an error, you should ``throw py::error_already_set();``, which allows
pybind11 to deal with the exception and pass it back to the Python interpreter.
(Another option is to call ``PyErr_Clear`` in the
`Python C-API <https://docs.python.org/3/c-api/exceptions.html#c.PyErr_Clear>`_
to clear the error. The Python error must be thrown or cleared, or Python/pybind11
will be left in an invalid state.)

There is also a special exception :class:`cast_error` that is thrown by
:func:`handle::call` when the input arguments cannot be converted to Python
objects.

Registering custom translators
==============================

If the default exception conversion policy described above is insufficient,
pybind11 also provides support for registering custom exception translators.
To register a simple exception conversion that translates a C++ exception into
a new Python exception using the C++ exception's ``what()`` method, a helper
function is available:

.. code-block:: cpp

    py::register_exception<CppExp>(module, "PyExp");

This call creates a Python exception class with the name ``PyExp`` in the given
module and automatically converts any encountered exceptions of type ``CppExp``
into Python exceptions of type ``PyExp``.

When more advanced exception translation is needed, the function
``py::register_exception_translator(translator)`` can be used to register
functions that can translate arbitrary exception types (and which may include
additional logic to do so).  The function takes a stateless callable (e.g.  a
function pointer or a lambda function without captured variables) with the call
signature ``void(std::exception_ptr)``.

When a C++ exception is thrown, the registered exception translators are tried
in reverse order of registration (i.e. the last registered translator gets the
first shot at handling the exception).

Inside the translator, ``std::rethrow_exception`` should be used within
a try block to re-throw the exception.  One or more catch clauses to catch
the appropriate exceptions should then be used with each clause using
``PyErr_SetString`` to set a Python exception or ``ex(string)`` to set
the python exception to a custom exception type (see below).

To declare a custom Python exception type, declare a ``py::exception`` variable
and use this in the associated exception translator (note: it is often useful
to make this a static declaration when using it inside a lambda expression
without requiring capturing).


The following example demonstrates this for a hypothetical exception classes
``MyCustomException`` and ``OtherException``: the first is translated to a
custom python exception ``MyCustomError``, while the second is translated to a
standard python RuntimeError:

.. code-block:: cpp

    static py::exception<MyCustomException> exc(m, "MyCustomError");
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const MyCustomException &e) {
            exc(e.what());
        } catch (const OtherException &e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });

Multiple exceptions can be handled by a single translator, as shown in the
example above. If the exception is not caught by the current translator, the
previously registered one gets a chance.

If none of the registered exception translators is able to handle the
exception, it is handled by the default converter as described in the previous
section.

.. seealso::

    The file :file:`tests/test_exceptions.cpp` contains examples
    of various custom exception translators and custom exception types.

.. note::

    You must call either ``PyErr_SetString`` or a custom exception's call
    operator (``exc(string)``) for every exception caught in a custom exception
    translator.  Failure to do so will cause Python to crash with ``SystemError:
    error return without exception set``.

    Exceptions that you do not plan to handle should simply not be caught, or
    may be explicitly (re-)thrown to delegate it to the other,
    previously-declared existing exception translators.

.. _unraisable_exceptions:

Handling unraisable exceptions
==============================

If a Python function invoked from a C++ destructor or any function marked
``noexcept(true)`` (collectively, "noexcept functions") throws an exception, there
is no way to propagate the exception, as such functions may not throw at
run-time.

Neither Python nor C++ allow exceptions raised in a noexcept function to propagate. In
Python, an exception raised in a class's ``__del__`` method is logged as an
unraisable error. In Python 3.8+, a system hook is triggered and an auditing
event is logged. In C++, ``std::terminate()`` is called to abort immediately.

Any noexcept function should have a try-catch block that traps
class:`error_already_set` (or any other exception that can occur). Note that pybind11
wrappers around Python exceptions such as :class:`pybind11::value_error` are *not*
Python exceptions; they are C++ exceptions that pybind11 catches and converts to
Python exceptions. Noexcept functions cannot propagate these exceptions either.
You can convert them to Python exceptions and then discard as unraisable.

.. code-block:: cpp

    void nonthrowing_func() noexcept(true) {
        try {
            // ...
        } catch (py::error_already_set &eas) {
            // Discard the Python error using Python APIs, using the C++ magic
            // variable __func__. Python already knows the type and value and of the
            // exception object.
            eas.discard_as_unraisable(__func__);
        } catch (const std::exception &e) {
            // Log and discard C++ exceptions.
            // (We cannot use discard_as_unraisable, since we have a generic C++
            // exception, not an exception that originated from Python.)
            third_party::log(e);
        }
    }
