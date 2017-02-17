STL containers
##############

Automatic conversion
====================

When including the additional header file :file:`pybind11/stl.h`, conversions
between ``std::vector<>``/``std::list<>``/``std::array<>``,
``std::set<>``/``std::unordered_set<>``, and
``std::map<>``/``std::unordered_map<>`` and the Python ``list``, ``set`` and
``dict`` data structures are automatically enabled. The types ``std::pair<>``
and ``std::tuple<>`` are already supported out of the box with just the core
:file:`pybind11/pybind11.h` header.

The major downside of these implicit conversions is that containers must be
converted (i.e. copied) on every Python->C++ and C++->Python transition, which
can have implications on the program semantics and performance. Please read the
next sections for more details and alternative approaches that avoid this.

.. note::

    Arbitrary nesting of any of these types is possible.

.. seealso::

    The file :file:`tests/test_python_types.cpp` contains a complete
    example that demonstrates how to pass STL data types in more detail.

.. _opaque:

Making opaque types
===================

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

.. code-block:: pycon

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
        .def(py::init<>())
        .def_readwrite("contents", &MyClass::contents);

In this case, properties can be read and written in their entirety. However, an
``append`` operation involving such a list type has no effect:

.. code-block:: pycon

   >>> m = MyClass()
   >>> m.contents = [5, 6]
   >>> print(m.contents)
   [5, 6]
   >>> m.contents.append(7)
   >>> print(m.contents)
   [5, 6]

Finally, the involved copy operations can be costly when dealing with very
large lists. To deal with all of the above situations, pybind11 provides a
macro named ``PYBIND11_MAKE_OPAQUE(T)`` that disables the template-based
conversion machinery of types, thus rendering them *opaque*. The contents of
opaque objects are never inspected or extracted, hence they *can* be passed by
reference. For instance, to turn ``std::vector<int>`` into an opaque type, add
the declaration

.. code-block:: cpp

    PYBIND11_MAKE_OPAQUE(std::vector<int>);

before any binding code (e.g. invocations to ``class_::def()``, etc.). This
macro must be specified at the top level (and outside of any namespaces), since
it instantiates a partial template overload. If your binding code consists of
multiple compilation units, it must be present in every file preceding any
usage of ``std::vector<int>``. Opaque types must also have a corresponding
``class_`` declaration to associate them with a name in Python, and to define a
set of available operations, e.g.:

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

The ability to expose STL containers as native Python objects is a fairly
common request, hence pybind11 also provides an optional header file named
:file:`pybind11/stl_bind.h` that does exactly this. The mapped containers try
to match the behavior of their native Python counterparts as much as possible.

The following example showcases usage of :file:`pybind11/stl_bind.h`:

.. code-block:: cpp

    // Don't forget this
    #include <pybind11/stl_bind.h>

    PYBIND11_MAKE_OPAQUE(std::vector<int>);
    PYBIND11_MAKE_OPAQUE(std::map<std::string, double>);

    // ...

    // later in binding code:
    py::bind_vector<std::vector<int>>(m, "VectorInt");
    py::bind_map<std::map<std::string, double>>(m, "MapStringDouble");

Please take a look at the :ref:`macro_notes` before using the
``PYBIND11_MAKE_OPAQUE`` macro.

.. seealso::

    The file :file:`tests/test_opaque_types.cpp` contains a complete
    example that demonstrates how to create and expose opaque types using
    pybind11 in more detail.

    The file :file:`tests/test_stl_binders.cpp` shows how to use the
    convenience STL container wrappers.
