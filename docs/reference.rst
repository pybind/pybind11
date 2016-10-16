.. _reference:

.. warning::

    Please be advised that the reference documentation discussing pybind11
    internals is currently incomplete. Please refer to the previous sections
    and the pybind11 header files for the nitty gritty details.

Reference
#########

Macros
======

.. function:: PYBIND11_PLUGIN(const char *name)

    This macro creates the entry point that will be invoked when the Python
    interpreter imports a plugin library. Please create a
    :class:`module` in the function body and return the pointer to its
    underlying Python object at the end.

    .. code-block:: cpp

        PYBIND11_PLUGIN(example) {
            pybind11::module m("example", "pybind11 example plugin");
            /// Set up bindings here
            return m.ptr();
        }

.. _core_types:

Convenience classes for arbitrary Python types
==============================================

Without reference counting
--------------------------

.. class:: handle

    The :class:`handle` class is a thin wrapper around an arbitrary Python
    object (i.e. a ``PyObject *`` in Python's C API). It does not perform any
    automatic reference counting and merely provides a basic C++ interface to
    various Python API functions.

.. seealso::

    The :class:`object` class inherits from :class:`handle` and adds automatic
    reference counting features.

.. function:: handle::handle()

    The default constructor creates a handle with a ``nullptr``-valued pointer.

.. function:: handle::handle(const handle&)

    Copy constructor

.. function:: handle::handle(PyObject *)

    Creates a :class:`handle` from the given raw Python object pointer.

.. function:: PyObject * handle::ptr() const

    Return the ``PyObject *`` underlying a :class:`handle`.

.. function:: const handle& handle::inc_ref() const

    Manually increase the reference count of the Python object. Usually, it is
    preferable to use the :class:`object` class which derives from
    :class:`handle` and calls this function automatically. Returns a reference
    to itself.

.. function:: const handle& handle::dec_ref() const

    Manually decrease the reference count of the Python object. Usually, it is
    preferable to use the :class:`object` class which derives from
    :class:`handle` and calls this function automatically. Returns a reference
    to itself.

.. function:: void handle::ref_count() const

    Return the object's current reference count

.. function:: handle handle::get_type() const

    Return a handle to the Python type object underlying the instance

.. function detail::accessor handle::operator[](handle key) const

    Return an internal functor to invoke the object's sequence protocol.
    Casting the returned ``detail::accessor`` instance to a :class:`handle` or
    :class:`object` subclass causes a corresponding call to ``__getitem__``.
    Assigning a :class:`handle` or :class:`object` subclass causes a call to
    ``__setitem__``.

.. function detail::accessor handle::operator[](const char *key) const

    See the above function (the only difference is that they key is provided as
    a string literal).

.. function detail::accessor handle::attr(handle key) const

    Return an internal functor to access the object's attributes.
    Casting the returned ``detail::accessor`` instance to a :class:`handle` or
    :class:`object` subclass causes a corresponding call to ``__getattr``.
    Assigning a :class:`handle` or :class:`object` subclass causes a call to
    ``__setattr``.

.. function detail::accessor handle::attr(const char *key) const

    See the above function (the only difference is that they key is provided as
    a string literal).

.. function operator handle::bool() const

    Return ``true`` when the :class:`handle` wraps a valid Python object.

.. function str handle::str() const

    Return a string representation of the object. This is analogous to
    the ``str()`` function in Python.

.. function:: template <typename T> T handle::cast() const

    Attempt to cast the Python object into the given C++ type. A
    :class:`cast_error` will be throw upon failure.

.. function:: template <typename ... Args> object handle::call(Args&&... args) const

    Assuming the Python object is a function or implements the ``__call__``
    protocol, ``call()`` invokes the underlying function, passing an arbitrary
    set of parameters. The result is returned as a :class:`object` and may need
    to be converted back into a Python object using :func:`handle::cast`.

    When some of the arguments cannot be converted to Python objects, the
    function will throw a :class:`cast_error` exception. When the Python
    function call fails, a :class:`error_already_set` exception is thrown.

With reference counting
-----------------------

.. class:: object : public handle

    Like :class:`handle`, the object class is a thin wrapper around an
    arbitrary Python object (i.e. a ``PyObject *`` in Python's C API). In
    contrast to :class:`handle`, it optionally increases the object's reference
    count upon construction, and it *always* decreases the reference count when
    the :class:`object` instance goes out of scope and is destructed. When
    using :class:`object` instances consistently, it is much easier to get
    reference counting right at the first attempt.

.. function:: object::object(const object &o)

    Copy constructor; always increases the reference count

.. function:: object::object(const handle &h, bool borrowed)

    Creates a :class:`object` from the given :class:`handle`. The reference
    count is only increased if the ``borrowed`` parameter is set to ``true``.

.. function:: object::object(PyObject *ptr, bool borrowed)

    Creates a :class:`object` from the given raw Python object pointer. The
    reference  count is only increased if the ``borrowed`` parameter is set to
    ``true``.

.. function:: object::object(object &&other)

    Move constructor; steals the object from ``other`` and preserves its
    reference count.

.. function:: handle object::release()

    Resets the internal pointer to ``nullptr`` without without decreasing the
    object's reference count. The function returns a raw handle to the original
    Python object.

.. function:: object::~object()

    Destructor, which automatically calls :func:`handle::dec_ref()`.

Convenience classes for specific Python types
=============================================


.. class:: module : public object

.. function:: module::module(const char *name, const char *doc = nullptr)

    Create a new top-level Python module with the given name and docstring

.. function:: module module::def_submodule(const char *name, const char *doc = nullptr)

    Create and return a new Python submodule with the given name and docstring.
    This also works recursively, i.e.

    .. code-block:: cpp

        pybind11::module m("example", "pybind11 example plugin");
        pybind11::module m2 = m.def_submodule("sub", "A submodule of 'example'");
        pybind11::module m3 = m2.def_submodule("subsub", "A submodule of 'example.sub'");

.. cpp:function:: template <typename Func, typename ... Extra> module& module::def(const char *name, Func && f, Extra && ... extra)

    Create Python binding for a new function within the module scope. ``Func``
    can be a plain C++ function, a function pointer, or a lambda function. For
    details on the ``Extra&& ... extra`` argument, see section :ref:`extras`.

.. _extras:

Passing extra arguments to the def function
===========================================

.. class:: arg

.. function:: arg::arg(const char *name)

.. function:: template <typename T> arg_v arg::operator=(T &&value)

.. class:: arg_v : public arg

    Represents a named argument with a default value

.. class:: sibling

    Used to specify a handle to an existing sibling function; used internally
    to implement function overloading in :func:`module::def` and
    :func:`class_::def`.

.. function:: sibling::sibling(handle handle)

.. class doc

    This is class is internally used by pybind11.

.. function:: doc::doc(const char *value)

    Create a new docstring with the specified value

.. class name

    This is class is internally used by pybind11.

.. function:: name::name(const char *value)

    Used to specify the function name

