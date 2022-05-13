Custom type casters
===================

Some applications may prefer custom type casters that convert between existing
Python types and C++ types, similar to the ``list`` ↔ ``std::vector``
and ``dict`` ↔ ``std::map`` conversions which are built into pybind11.
Implementing custom type casters is fairly advanced usage and requires
familiarity with the intricacies of the Python C API.

The following snippets demonstrate how this works for a very simple ``inty``
type that we want to be convertible to C++ from any Python type that provides
an ``__int__`` method, and is converted to a Python ``int`` when returned from
C++ to Python.

..
    PLEASE KEEP THE CODE BLOCKS IN SYNC WITH
        tests/test_docs_advanced_cast_custom.cpp
    Ideally, change the test, run pre-commit (incl. clang-format),
    then copy the changed code back here.

.. code-block:: cpp

    namespace user_space {

    struct inty {
        long long_value;
    };

    std::string to_string(const inty &s) { return std::to_string(s.long_value); }

    inty return_42() { return inty{42}; }

    } // namespace user_space

The necessary conversion routines are defined by a caster class, which
is then "plugged into" pybind11 using one of two alternative mechanisms.

Starting with the example caster class:

.. code-block:: cpp

    namespace user_space {

    struct inty_type_caster {
    public:
        // This macro establishes the name 'inty' in function signatures and declares a local variable
        // 'value' of type inty.
        PYBIND11_TYPE_CASTER(inty, pybind11::detail::const_name("inty"));

        // Python -> C++: convert a PyObject into an inty instance or return false upon failure. The
        // second argument indicates whether implicit conversions should be allowed.
        bool load(pybind11::handle src, bool) {
            // Extract PyObject from handle.
            PyObject *source = src.ptr();
            // Try converting into a Python integer value.
            PyObject *tmp = PyNumber_Long(source);
            if (!tmp) {
                return false;
            }
            // Now try to convert into a C++ int.
            value.long_value = PyLong_AsLong(tmp);
            Py_DECREF(tmp);
            // Ensure PyLong_AsLong succeeded (to catch out-of-range errors etc).
            if (PyErr_Occurred()) {
                PyErr_Clear();
                return false;
            }
            return true;
        }

        // C++ -> Python: convert an inty instance into a Python object. The second and third arguments
        // are used to indicate the return value policy and parent object (for
        // return_value_policy::reference_internal) and are often ignored by custom casters.
        static pybind11::handle
        cast(inty src, pybind11::return_value_policy /* policy */, pybind11::handle /* parent */) {
            return PyLong_FromLong(src.long_value);
        }
    };

    } // namespace user_space

.. note::

    A caster class using ``PYBIND11_TYPE_CASTER(T, ...)`` requires
    that ``T`` is default-constructible (``value`` is first default constructed
    and then ``load()`` assigns to it). It is possible but more involved to define
    a caster class for types that are not default-constructible.

The caster class defined above can be plugged into pybind11 in two ways:

* Starting with pybind11 v2.10, a new — and the recommended — alternative is to *declare* a
  function named ``pybind11_select_caster``:

  .. code-block:: cpp

    namespace user_space {

    inty_type_caster pybind11_select_caster(inty *);

    } // namespace user_space

  The argument is a *pointer* to the C++ type, the return type is the caster type.
  This function has no implementation! Its only purpose is to associate the C++ type
  with its caster class. pybind11 exploits C++ Argument Dependent Lookup
  (`ADL <https://en.cppreference.com/w/cpp/language/adl>`_)
  to discover the association.

  Note that ``pybind11_select_caster`` can alternatively be declared as a ``friend``
  function of the C++ type, if that is practical and preferred:

  .. code-block:: cpp

    struct inty_type_caster;

    struct inty {
        ...
        friend inty_type_caster pybind11_select_caster(inty *);
    };

* An older alternative is to specialize the ``pybind11::detail::type_caster<T>`` template.
  Although the ``detail`` namespace is involved, adding a ``type_caster`` specialization
  is explicitly allowed:

  .. code-block:: cpp

    namespace pybind11 {
    namespace detail {
    template <>
    struct type_caster<user_space::inty> : user_space::inty_type_caster {};
    } // namespace detail
    } // namespace pybind11

  .. note::
      ``type_caster` specializations may be full (as in this simple example) or partial.

.. warning::
    With either alternative, for a given type ``T``, the ``pybind11_select_caster``
    declaration or ``type_caster`` specialization must be consistent across all compilation
    units of a Python extension module, to satisfy the C++ One Definition Rule
    (`ODR <https://en.cppreference.com/w/cpp/language/definition>`_).
