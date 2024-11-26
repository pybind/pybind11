Custom type casters
===================

Some applications may prefer custom type casters that convert between existing
Python types and C++ types, similar to the ``list`` ↔ ``std::vector``
and ``dict`` ↔ ``std::map`` conversions which are built into pybind11.
Implementing custom type casters is fairly advanced usage and requires
familiarity with the intricacies of the Python C API.
You can refer to the `Python/C API Reference Manual <https://docs.python.org/3/c-api/index.html>`_
for more information.

The following snippets demonstrate how this works for a very simple ``Point2D`` type.
We want this type to be convertible to C++ from Python types implementing the
``Sequence`` protocol and having two elements of type ``float``.
When returned from C++ to Python, it should be converted to a Python ``tuple[float, float]``.
For this type we could provide Python bindings for different arithmetic functions implemented
in C++ (here demonstrated by a simple ``negate`` function).

..
    PLEASE KEEP THE CODE BLOCKS IN SYNC WITH
        tests/test_docs_advanced_cast_custom.cpp
        tests/test_docs_advanced_cast_custom.py
    Ideally, change the test, run pre-commit (incl. clang-format),
    then copy the changed code back here.
    Also use TEST_SUBMODULE in tests, but PYBIND11_MODULE in docs.

.. code-block:: cpp

    namespace user_space {

    struct Point2D {
        double x;
        double y;
    };

    Point2D negate(const Point2D &point) { return Point2D{-point.x, -point.y}; }

    } // namespace user_space


The following Python snippet demonstrates the intended usage of ``negate`` from the Python side:

.. code-block:: python

    from my_math_module import docs_advanced_cast_custom as m

    point1 = [1.0, -1.0]
    point2 = m.negate(point1)
    assert point2 == (-1.0, 1.0)

To register the necessary conversion routines, it is necessary to add an
instantiation of the ``pybind11::detail::type_caster<T>`` template.
Although this is an implementation detail, adding an instantiation of this
type is explicitly allowed.

.. code-block:: cpp

    namespace pybind11 {
    namespace detail {

    template <>
    struct type_caster<user_space::Point2D> {
        // This macro inserts a lot of boilerplate code and sets the default type hint to `tuple`
        PYBIND11_TYPE_CASTER(user_space::Point2D, const_name("tuple"));
        // `arg_name` and `return_name` may optionally be used to specify type hints separately for
        // arguments and return values.
        // The signature of our identity function would then look like:
        // `identity(Sequence[float]) -> tuple[float, float]`
        static constexpr auto arg_name = const_name("Sequence[float]");
        static constexpr auto return_name = const_name("tuple[float, float]");

        // C++ -> Python: convert `Point2D` to `tuple[float, float]`. The second and third arguments
        // are used to indicate the return value policy and parent object (for
        // return_value_policy::reference_internal) and are often ignored by custom casters.
        static handle cast(const user_space::Point2D &number, return_value_policy, handle) {
            // Convert x and y components to python float
            auto *x = PyFloat_FromDouble(number.x);
            auto *y = PyFloat_FromDouble(number.y);
            // Check if conversion was successful otherwise clean up references and return null
            if (!x || !y) {
                Py_XDECREF(x);
                Py_XDECREF(y);
                return nullptr;
            }
            // Create tuple from x and y
            auto t = PyTuple_Pack(2, x, y);
            // Decrement references (the tuple now owns x an y)
            Py_DECREF(x);
            Py_DECREF(y);
            return t;
        }

        // Python -> C++: convert a `PyObject` into a `Point2D` and return false upon failure. The
        // second argument indicates whether implicit conversions should be allowed.
        bool load(handle src, bool) {
            // Check if handle is valid Sequence of length 2
            if (!src || PySequence_Check(src.ptr()) == 0 || PySequence_Length(src.ptr()) != 2) {
                return false;
            }
            auto *x = PySequence_GetItem(src.ptr(), 0);
            auto *y = PySequence_GetItem(src.ptr(), 1);
            // Check if values are float or int (both are allowed with float as type hint)
            if (!x || !(PyFloat_Check(x) || PyLong_Check(x)) || !y
                || !(PyFloat_Check(y) || PyLong_Check(y))) {
                Py_XDECREF(x);
                Py_XDECREF(y);
                return false;
            }
            // value is a default constructed Point2D
            value.x = PyFloat_AsDouble(x);
            value.y = PyFloat_AsDouble(y);
            Py_DECREF(x);
            Py_DECREF(y);
            if ((value.x == -1.0 || value.y == -1.0) && PyErr_Occurred()) {
                PyErr_Clear();
                return false;
            }
            return true;
        }
    };

    } // namespace detail
    } // namespace pybind11

    // Bind the negate function
    PYBIND11_MODULE(docs_advanced_cast_custom, m) { m.def("negate", user_space::negate); }

.. note::

    A ``type_caster<T>`` defined with ``PYBIND11_TYPE_CASTER(T, ...)`` requires
    that ``T`` is default-constructible (``value`` is first default constructed
    and then ``load()`` assigns to it).

.. warning::

    When using custom type casters, it's important to declare them consistently
    in every compilation unit of the Python extension module to satisfy the C++ One Definition Rule
    (`ODR <https://en.cppreference.com/w/cpp/language/definition>`_).. Otherwise,
    undefined behavior can ensue.

.. note::

    Using the type hint ``Sequence[float]`` signals to static type checkers, that not only tuples may be
    passed, but any type implementing the Sequence protocol, e.g., ``list[float]``.
    Unfortunately, that loses the length information ``tuple[float, float]`` provides.
    One way of still providing some length information in type hints is using ``typing.Annotated``, e.g.,
    ``Annotated[Sequence[float], 2]``, or further add libraries like
    `annotated-types <https://github.com/annotated-types/annotated-types>`_.
