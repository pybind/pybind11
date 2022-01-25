Custom type casters
===================

In very rare cases, applications may require custom type casters that cannot be
expressed using the abstractions provided by pybind11, thus requiring raw
Python C API calls. This is fairly advanced usage and should only be pursued by
experts who are familiar with the intricacies of Python reference counting.

The following snippets demonstrate how this works for a very simple ``inty``
type that that should be convertible from Python types that provide a
``__int__(self)`` method.

.. code-block:: cpp

    struct inty { long long_value; };

    void print(inty s) {
        std::cout << s.long_value << std::endl;
    }

The following Python snippet demonstrates the intended usage from the Python side:

.. code-block:: python

    class A:
        def __int__(self):
            return 123


    from example import print

    print(A())

To register the necessary conversion routines, it is necessary to define a
caster class, and register it with the other pybind11 casters:

.. code-block:: cpp

    struct inty_type_caster {
    public:
        /**
         * This macro establishes the name 'inty' in
         * function signatures and declares a local variable
         * 'value' of type inty
         */
        PYBIND11_TYPE_CASTER(inty, const_name("inty"));

        /**
         * Conversion part 1 (Python->C++): convert a PyObject into a inty
         * instance or return false upon failure. The second argument
         * indicates whether implicit conversions should be applied.
         */
        bool load(handle src, bool) {
            /* Extract PyObject from handle */
            PyObject *source = src.ptr();
            /* Try converting into a Python integer value */
            PyObject *tmp = PyNumber_Long(source);
            if (!tmp)
                return false;
            /* Now try to convert into a C++ int */
            value.long_value = PyLong_AsLong(tmp);
            Py_DECREF(tmp);
            /* Ensure return code was OK (to avoid out-of-range errors etc) */
            return !(value.long_value == -1 && !PyErr_Occurred());
        }

        /**
         * Conversion part 2 (C++ -> Python): convert an inty instance into
         * a Python object. The second and third arguments are used to
         * indicate the return value policy and parent object (for
         * ``return_value_policy::reference_internal``) and are generally
         * ignored by implicit casters.
         */
        static handle cast(inty src, return_value_policy /* policy */, handle /* parent */) {
            return PyLong_FromLong(src.long_value);
        }
    };

.. note::

    A caster class defined with ``PYBIND11_TYPE_CASTER(T, ...)`` requires
    that ``T`` is default-constructible (``value`` is first default constructed
    and then ``load()`` assigns to it).

The caster defined above must be registered with pybind11.
There are two ways to do it:

* As an instantiation of the ``pybind11::detail::type_caster<T>`` template.
  Although this is an implementation detail, adding an instantiation of this
  type is explicitly allowed:

  .. code-block:: cpp

      namespace pybind11 { namespace detail {
          template <> struct type_caster<inty> : inty_type_caster {};
      }} // namespace pybind11::detail

  .. warning::

      When using this method, it's important to declare them consistently
      in every compilation unit of the Python extension module. Otherwise,
      undefined behavior can ensue.

* The preferred method is to *declare* a function named
  ``pybind11_select_caster``, its only purpose is to associate the C++ type
  with its caster class:

  .. code-block:: cpp

      inty_type_caster pybind11_select_caster(inty*);

  The argument is a *pointer* to the C++ type, the return type is the
  caster type. This function has no implementation!
