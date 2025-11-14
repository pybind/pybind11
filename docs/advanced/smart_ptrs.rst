.. _py_class_holder:

Smart pointers & ``py::class_``
###############################

The binding generator for classes, ``py::class_``, can be passed a template
type that denotes a special *holder* type that is used to manage references to
the object. If no such holder type template argument is given, the default for
a type ``T`` is ``std::unique_ptr<T>``.

.. note::

    A ``py::class_`` for a given C++ type ``T`` — and all its derived types —
    can only use a single holder type.


.. _smart_holder:

``py::smart_holder``
====================

Starting with pybind11v3, ``py::smart_holder`` is built into pybind11. It is
the recommended ``py::class_`` holder for most situations. However, for
backward compatibility it is **not** the default holder, and there are no
plans to make it the default holder in the future.

It is extremely easy to use the safer and more versatile ``py::smart_holder``:
simply add ``py::smart_holder`` to ``py::class_``:

* ``py::class_<T>`` to

* ``py::class_<T, py::smart_holder>``.

.. note::

    A shorthand, ``py::classh<T>``, is provided for
    ``py::class_<T, py::smart_holder>``. The ``h`` in ``py::classh`` stands
    for **smart_holder** but is shortened for brevity, ensuring it has the
    same number of characters as ``py::class_``. This design choice facilitates
    easy experimentation with ``py::smart_holder`` without introducing
    distracting whitespace noise in diffs.

The ``py::smart_holder`` functionality includes the following:

* Support for **two-way** Python/C++ conversions for both
  ``std::unique_ptr<T>`` and ``std::shared_ptr<T>`` **simultaneously**.

* Passing a Python object back to C++ via ``std::unique_ptr<T>``, safely
  **disowning** the Python object.

* Safely passing "trampoline" objects (objects with C++ virtual function
  overrides implemented in Python, see :ref:`overriding_virtuals`) via
  ``std::unique_ptr<T>`` or ``std::shared_ptr<T>`` back to C++:
  associated Python objects are automatically kept alive for the lifetime
  of the smart-pointer.

* Full support for ``std::enable_shared_from_this`` (`cppreference
  <http://en.cppreference.com/w/cpp/memory/enable_shared_from_this>`_).


``std::unique_ptr``
===================

This is the default ``py::class_`` holder and works as expected in
most situations. However, handling base-and-derived classes involves a
``reinterpret_cast``, which is, strictly speaking, undefined behavior.
Also note that the ``std::unique_ptr`` holder only supports passing a
``std::unique_ptr`` from C++ to Python, but not the other way around.
For example, the following code works as expected with ``py::class_<Example>``:

.. code-block:: cpp

    std::unique_ptr<Example> create_example() { return std::unique_ptr<Example>(new Example()); }

.. code-block:: cpp

    m.def("create_example", &create_example);

However, this will fail with ``py::class_<Example>`` (but works with
``py::class_<Example, py::smart_holder>``):

.. code-block:: cpp

    void do_something_with_example(std::unique_ptr<Example> ex) { ... }

.. note::

    The ``reinterpret_cast`` mentioned above is `here
    <https://github.com/pybind/pybind11/blob/30eb39ed79d1e2eeff15219ac00773034300a5e6/include/pybind11/cast.h#L235>`_.
    For completeness: The same cast is also applied to ``py::smart_holder``,
    but that is safe, because ``py::smart_holder`` is not templated.


``std::shared_ptr``
===================

It is possible to use ``std::shared_ptr`` as the holder, for example:

.. code-block:: cpp

    py::class_<Example, std::shared_ptr<Example> /* <- holder type */>(m, "Example");

Compared to using ``py::class_<Example, py::smart_holder>``, there are two noteworthy disadvantages:

* Because a ``py::class_`` for a given C++ type ``T`` can only use a
  single holder type, ``std::unique_ptr<T>`` cannot even be passed from C++
  to Python. This will become apparent only at runtime, often through a
  segmentation fault.

* Similar to the ``std::unique_ptr`` holder, the handling of base-and-derived
  classes involves a ``reinterpret_cast`` that has strictly speaking undefined
  behavior, although it works as expected in most situations.


.. _smart_pointers:

Custom smart pointers
=====================

For custom smart pointers (e.g. ``c10::intrusive_ptr`` in pytorch), transparent
conversions can be enabled using a macro invocation similar to the following.
It must be declared at the top namespace level before any binding code:

.. code-block:: cpp

    PYBIND11_DECLARE_HOLDER_TYPE(T, SmartPtr<T>)

The first argument of :func:`PYBIND11_DECLARE_HOLDER_TYPE` should be a
placeholder name that is used as a template parameter of the second argument.
Thus, feel free to use any identifier, but use it consistently on both sides;
also, don't use the name of a type that already exists in your codebase.

The macro also accepts a third optional boolean parameter that is set to false
by default. Specify

.. code-block:: cpp

    PYBIND11_DECLARE_HOLDER_TYPE(T, SmartPtr<T>, true)

if ``SmartPtr<T>`` can always be initialized from a ``T*`` pointer without the
risk of inconsistencies (such as multiple independent ``SmartPtr`` instances
believing that they are the sole owner of the ``T*`` pointer). A common
situation where ``true`` should be passed is when the ``T`` instances use
*intrusive* reference counting.

Please take a look at the :ref:`macro_notes` before using this feature.

By default, pybind11 assumes that your custom smart pointer has a standard
interface, i.e. provides a ``.get()`` member function to access the underlying
raw pointer. If this is not the case, pybind11's ``holder_helper`` must be
specialized:

.. code-block:: cpp

    // Always needed for custom holder types
    PYBIND11_DECLARE_HOLDER_TYPE(T, SmartPtr<T>)

    // Only needed if the type's `.get()` goes by another name
    namespace PYBIND11_NAMESPACE { namespace detail {
        template <typename T>
        struct holder_helper<SmartPtr<T>> { // <-- specialization
            static const T *get(const SmartPtr<T> &p) { return p.getPointer(); }
        };
    }}

The above specialization informs pybind11 that the custom ``SmartPtr`` class
provides ``.get()`` functionality via ``.getPointer()``.

.. note::

    The two noteworthy disadvantages mentioned under the ``std::shared_ptr``
    section apply similarly to custom smart pointer holders, but there is no
    established safe alternative in this case.

.. seealso::

    The file :file:`tests/test_smart_ptr.cpp` contains a complete example
    that demonstrates how to work with custom reference-counting holder types
    in more detail.
