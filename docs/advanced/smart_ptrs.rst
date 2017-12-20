.. _holders:

Smart pointers and holders
##########################

Holders
=======

The binding generator for classes, :class:`class_`, can be passed a template
type that denotes a special *holder* type that is used to manage references to
the object.  If no such holder type template argument is given, the default for
a type named ``Type`` is ``std::unique_ptr<Type>``, which means that the object
is deallocated when Python's reference count goes to zero. It is possible to switch to other types of reference counting wrappers or smart
pointers, which is useful in codebases that rely on them, such as ``std::shared_ptr<Type>``, or even a custom type.

std::unique_ptr
===============

Given a class ``Example`` with Python bindings, it's possible to return
instances wrapped in C++11 unique pointers, like so

.. code-block:: cpp

    std::unique_ptr<Example> create_example() { return std::unique_ptr<Example>(new Example()); }

.. code-block:: cpp

    m.def("create_example", &create_example);

In other words, there is nothing special that needs to be done.

.. _unique_ptr_ownership:

Transferring ownership
----------------------

It is also possible to pass ``std::unique_ptr<Type>`` as a function
argument, or use ``py::cast<unique_ptr<Type>>(std::move(obj))``. Note that this tells pybind11 to not manage the memory for this object, and delegate that to ``std::unique_ptr<Type>``.
For instance, the following function signature can be processed by pybind11:

.. code-block:: cpp

    void do_something_with_example(std::unique_ptr<Example> ex) { ... }

The above signature does imply that Python needs to give up ownership of an
object that is passed to this function. There are two ways to do this:

1.  Simply pass the object in. The reference count of the object can be greater than one (non-unique) when passing the object in. **However**, you *must* ensure that the object has only **one** reference when C++ (which owns the C++ object).

    To expand on this, when transferring ownership for ``std::unique_ptr``, this means that Pybind11 no longer owns the reference, which means that if C++ lets the ``std::unique_ptr`` destruct but if there is a dangling reference in Python, then you will encounter undefined behavior.

    Examples situations:

    * The C++ function is terminal (i.e. will destroy the object once it completes). This is generally not an issue unless a Python portion of the object has a non-trivial ``__del__`` method.
    * The Python object is passed to a C++ container which only tracks ``std::unique_ptr<Type>``. If the container goes out of scope, the Python object reference will be invalid.

    .. note::

        For polymorphic types that inherit from :class:`py::wrapper`, ``pybind11`` *can* warn about these situations.
        You may enable this behavior with ``#define PYBIND11_WARN_DANGLING_UNIQUE_PYREF``. This will print a warning to ``std::err`` if this case is detected.

2.  Pass a Python "move container" (a mutable object that can "release" the reference to the object). This can be a single-item list, or any Python class / instance that has the field ``_is_move_container = True`` and has a ``release()`` function.

    .. note::

        When using a move container, this expects that the provided object is a **unique** reference, or will throw an error otherwise. This is a little more verbose, but will make debugging *much* easier.

    As an example in C++:

    .. code-block:: cpp

        void terminal_func(std::unique_ptr<Example> obj) {
            obj.do_something();  // `obj` will be destroyed when the function exits.
        }

        // Binding
        py::class_<Example> example(m, "Example");
        m.def("terminal_func", &terminal_func);

    In Python, say you would normally do this:

    .. code-block:: pycon

        >>> obj = Example()
        >>> terminal_func(obj)

    As mentioned in the comment, you *must* ensure that `obj` is not used past this invocation, as the underlying data has been destroyed. To be more careful, you may "move" the object. The following will throw an error:

    .. code-block:: pycon

        >>> obj = Example()
        >>> terminal_func([obj])

    However, this will work, using a "move" container:

    .. code-block:: pycon

        >>> obj = Example()
        >>> obj_move = [obj]
        >>> del obj
        >>> terminal_func(obj_move)
        >>> print(obj_move)  # Reference will have been removed.
        [None]

    or even:

    .. code-block:: pycon

        >>> terminal_func([Example()])

    .. note::

        ``terminal_func(Example())`` also works, but still leaves a dangling reference, which is only a problem if it is polymorphic and has a non-trivial ``__del__`` method.

    .. warning::

        This reference counting mechanism is **not** robust aganist cyclic references. If you need some sort of cyclic reference, *please* consider using ``weakref.ref`` in Python.

std::shared_ptr
===============

If you have an existing code base with ``std::shared_ptr``, or you wish to enable reference counting in C++ as well, then you may use this type as a holder.
As an example, the following snippet causes ``std::shared_ptr`` to be used instead.

.. code-block:: cpp

    py::class_<Example, std::shared_ptr<Example> /* <- holder type */> obj(m, "Example");

Note that any particular class can only be associated with a single holder type.

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

.. seealso::

    While ownership transfer is generally not an issue with ``std::shared_ptr<Type>``, it becomes an issue when an instance of a Python subclass of a pybind11 class is effectively managed by C++ (e.g. all live references to the object are from C++, and all reference in Python have "died").

    See :ref:`virtual_inheritance_lifetime` for more information.

.. _smart_pointers:

Custom smart pointers
=====================

pybind11 supports ``std::unique_ptr`` and ``std::shared_ptr`` right out of the
box. For any other custom smart pointer, transparent conversions can be enabled
using a macro invocation similar to the following. It must be declared at the
top namespace level before any binding code:

.. code-block:: cpp

    PYBIND11_DECLARE_HOLDER_TYPE(T, SmartPtr<T>);

The first argument of :func:`PYBIND11_DECLARE_HOLDER_TYPE` should be a
placeholder name that is used as a template parameter of the second argument.
Thus, feel free to use any identifier, but use it consistently on both sides;
also, don't use the name of a type that already exists in your codebase.

The macro also accepts a third optional boolean parameter that is set to false
by default. Specify

.. code-block:: cpp

    PYBIND11_DECLARE_HOLDER_TYPE(T, SmartPtr<T>, true);

if ``SmartPtr<T>`` can always be initialized from a ``T*`` pointer without the
risk of inconsistencies (such as multiple independent ``SmartPtr`` instances
believing that they are the sole owner of the ``T*`` pointer). A common
situation where ``true`` should be passed is when the ``T`` instances use
*intrusive* reference counting.

Please take a look at the :ref:`macro_notes` before using this feature.

By default, pybind11 assumes that your custom smart pointer has a standard
interface, i.e. provides a ``.get()`` member function to access the underlying
raw pointer, and a ``.release()`` member function for move-only holders. If this is not the case, pybind11's ``holder_helper`` must be
specialized:

.. code-block:: cpp

    // Always needed for custom holder types
    PYBIND11_DECLARE_HOLDER_TYPE(T, SmartPtr<T>);

    // Only needed if the type's `.get()` goes by another name
    namespace pybind11 { namespace detail {
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

.. warning::

    Holder type conversion (see :ref:`smart_ptrs_casting`) and advanced ownership transfer (see :ref:`virtual_inheritance_lifetime`) is **not** supported for custom shared pointer types, due to constraints on dynamic type erasure.

.. _smart_ptrs_casting:

Casting smart pointers
======================

As shown in the :ref:`conversion_table`, you may cast to any of the available holders (e.g. ``py::cast<std::shared_ptr<Type>>(obj)``) that can properly provide access to the underlying holder.

``pybind11`` will raise an error if there is an incompatible cast. You may of course cast to the exact same holder type. You may also move a ``std::unique_ptr<Type>`` into a ``std::shared_ptr<Type>``, as this is allowed. **However**, you may not convert a ``std::shared_ptr<Type>`` to a ``std::unique_ptr<Type>`` as you cannot release an object that is managed by ``std::shared_ptr<Type>``.

Additionally, conversion to ``std::unique_ptr<Type, Deleter>`` is not supported if ``Deleter`` is not ``std::default_deleter<Type>``.

Conversion to a different custom smart pointer is not supported.
