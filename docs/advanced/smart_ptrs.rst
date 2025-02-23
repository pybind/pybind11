Smart pointers & ``py::class_``
###############################

The binding generator for classes, ``py::class_``, can be passed a template
type that denotes a special *holder* type that is used to manage references to
the object.  If no such holder type template argument is given, the default for
a type ``T`` is ``std::unique_ptr<T>``.

``py::smart_holder``
====================

Starting with pybind11v3, ``py::smart_holder`` is built into pybind11. It is
the recommended ``py::class_`` holder for all situations, but it is **not**
the default holder, and there is no intent to make it the default holder in
the future, based on the assumption that this would cause more disruption
than it is worth.

It is extremely easy to change existing pybind11 client code to use the safer
and more versatile ``py::smart_holder``. For a given C++ type ``T``, simply
change

* ``py::class_<T>`` to
* ``py::classh<T>``

.. note::

    ``py::classh<T>`` is simply a shortcut for ``py::class_<T, py::smart_holder>``.

The ``py::classh<T>`` functionality includes

* support for **two-way** Python/C++ conversions for both
  ``std::unique_ptr<T>`` and ``std::shared_ptr<T>`` **simultaneously**.
  — In contrast, ``py::class_<T>`` only supports one-way C++-to-Python
  conversions for ``std::unique_ptr<T>``, or alternatively two-way
  Python/C++ conversions for ``std::shared_ptr<T>``, which then excludes
  the one-way C++-to-Python ``std::unique_ptr<T>`` conversions (this manifests
  itself through undefined runtime behavior, often a segmentation fault
  or double free).

* passing a Python object back to C++ via ``std::unique_ptr<T>``, safely
  **disowning** the Python object.

* safely passing `"trampoline"
  <https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python>`_
  objects (objects with C++ virtual function overrides implemented in
  Python) via ``std::unique_ptr<T>`` or ``std::shared_ptr<T>`` back to C++:
  associated Python objects are automatically kept alive for the lifetime
  of the smart-pointer.

TODO(rwgk): Move to classes.rst

A pybind11 `"trampoline"
<https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python>`_
is a C++ helper class with virtual function overrides that transparently
call back from C++ into Python. To enable safely passing a ``std::unique_ptr``
to a trampoline object between Python and C++, the trampoline class must
inherit from ``py::trampoline_self_life_support``, for example:

.. code-block:: cpp

   class PyAnimal : public Animal, public py::trampoline_self_life_support {
       ...
   };

A fairly minimal but complete example is :file:`tests/test_class_sh_trampoline_unique_ptr.cpp`.


``std::unique_ptr``
===================

This is the default ``py::class_`` holder and works as expected in most
situations. However, note that the handling of base-and-derived classes
involves a ``reinterpret_cast`` that has strictly speaking undefined
behavior. Also note that the ``std::unique_ptr`` holder only support passing
a ``std::unique_ptr`` from C++ to Python, but not the other way around. For
example, this code will work as expected when using ``py::class_<Example>``:

.. code-block:: cpp

    std::unique_ptr<Example> create_example() { return std::unique_ptr<Example>(new Example()); }

.. code-block:: cpp

    m.def("create_example", &create_example);

However, this will fail with ``py::class_<Example>`` (but work with
``py::classh<Example>``):

.. code-block:: cpp

    void do_something_with_example(std::unique_ptr<Example> ex) { ... }


``std::shared_ptr``
===================

It is possible to use ``std::shared_ptr`` as the holder, for example:

.. code-block:: cpp

    py::class_<Example, std::shared_ptr<Example> /* <- holder type */>(m, "Example");

Compared to using ``py::classh``, there are two noteworthy disadvantages:

* A ``py::class_`` for any particular C++ type ``T`` (and all its derived types)
  can only use a single holder type. Therefore, ``std::unique_ptr<T>``
  cannot even be passed from C++ to Python if the ``std::shared_ptr<T>`` holder
  is used. This will become apparent only at runtime, often through a
  segmentation fault or double free.

* Similar to the ``std::unique_ptr`` holder, the handling of base-and-derived
  classes involves a ``reinterpret_cast`` that has strictly speaking undefined
  behavior, although it works as expected in most situations.


.. _smart_pointers:

Custom smart pointers
=====================

For custom smart pointer, transparent conversions can be enabled
using a macro invocation similar to the following. It must be declared at the
top namespace level before any binding code:

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

.. seealso::

    The file :file:`tests/test_smart_ptr.cpp` contains a complete example
    that demonstrates how to work with custom reference-counting holder types
    in more detail.


Be careful to not undermine automatic lifetime management
=========================================================

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

    PYBIND11_MODULE(example, m) {
        py::class_<Child, std::shared_ptr<Child>>(m, "Child");

        py::class_<Parent, std::shared_ptr<Parent>>(m, "Parent")
           .def(py::init<>())
           .def("get_child", &Parent::get_child);
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
