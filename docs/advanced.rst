.. _advanced:

Advanced topics
###############

For brevity, the rest of this chapter assumes that the following two lines are
present:

.. code-block:: cpp

    #include <pybind11/pybind11.h>

    namespace py = pybind11;

Operator overloading
====================

Suppose that we're given the following ``Vector2`` class with a vector addition
and scalar multiplication operation, all implemented using overloaded operators
in C++.

.. code-block:: cpp

    class Vector2 {
    public:
        Vector2(float x, float y) : x(x), y(y) { }

        std::string toString() const { return "[" + std::to_string(x) + ", " + std::to_string(y) + "]"; }

        Vector2 operator+(const Vector2 &v) const { return Vector2(x + v.x, y + v.y); }
        Vector2 operator*(float value) const { return Vector2(x * value, y * value); }
        Vector2& operator+=(const Vector2 &v) { x += v.x; y += v.y; return *this; }
        Vector2& operator*=(float v) { x *= v; y *= v; return *this; }

        friend Vector2 operator*(float f, const Vector2 &v) { return Vector2(f * v.x, f * v.y); }

    private:
        float x, y;
    };

The following snippet shows how the above operators can be conveniently exposed
to Python.

.. code-block:: cpp

    #include <pybind11/operators.h>

    PYBIND11_PLUGIN(example) {
        py::module m("example", "pybind11 example plugin");

        py::class_<Vector2>(m, "Vector2")
            .def(py::init<float, float>())
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def(py::self *= float())
            .def(float() * py::self)
            .def("__repr__", &Vector2::toString);

        return m.ptr();
    }

Note that a line like

.. code-block:: cpp

            .def(py::self * float())

is really just short hand notation for

.. code-block:: cpp

    .def("__mul__", [](const Vector2 &a, float b) {
        return a * b;
    })

This can be useful for exposing additional operators that don't exist on the
C++ side, or to perform other types of customization.

.. note::

    To use the more convenient ``py::self`` notation, the additional
    header file :file:`pybind11/operators.h` must be included.

.. seealso::

    The file :file:`example/example3.cpp` contains a complete example that
    demonstrates how to work with overloaded operators in more detail.

Callbacks and passing anonymous functions
=========================================

The C++11 standard brought lambda functions and the generic polymorphic
function wrapper ``std::function<>`` to the C++ programming language, which
enable powerful new ways of working with functions. Lambda functions come in
two flavors: stateless lambda function resemble classic function pointers that
link to an anonymous piece of code, while stateful lambda functions
additionally depend on captured variables that are stored in an anonymous
*lambda closure object*.

Here is a simple example of a C++ function that takes an arbitrary function
(stateful or stateless) with signature ``int -> int`` as an argument and runs
it with the value 10.

.. code-block:: cpp

    int func_arg(const std::function<int(int)> &f) {
        return f(10);
    }

The example below is more involved: it takes a function of signature ``int -> int``
and returns another function of the same kind. The return value is a stateful
lambda function, which stores the value ``f`` in the capture object and adds 1 to
its return value upon execution.

.. code-block:: cpp

    std::function<int(int)> func_ret(const std::function<int(int)> &f) {
        return [f](int i) {
            return f(i) + 1;
        };
    }

After including the extra header file :file:`pybind11/functional.h`, it is almost
trivial to generate binding code for both of these functions.

.. code-block:: cpp

    #include <pybind11/functional.h>

    PYBIND11_PLUGIN(example) {
        py::module m("example", "pybind11 example plugin");

        m.def("func_arg", &func_arg);
        m.def("func_ret", &func_ret);

        return m.ptr();
    }

The following interactive session shows how to call them from Python.

.. code-block:: python

    $ python
    >>> import example
    >>> def square(i):
    ...     return i * i
    ...
    >>> example.func_arg(square)
    100L
    >>> square_plus_1 = example.func_ret(square)
    >>> square_plus_1(4)
    17L
    >>>

.. note::

    This functionality is very useful when generating bindings for callbacks in
    C++ libraries (e.g. a graphical user interface library).

    The file :file:`example/example5.cpp` contains a complete example that
    demonstrates how to work with callbacks and anonymous functions in more detail.

.. warning::

    Keep in mind that passing a function from C++ to Python (or vice versa)
    will instantiate a piece of wrapper code that translates function
    invocations between the two languages. Copying the same function back and
    forth between Python and C++ many times in a row will cause these wrappers
    to accumulate, which can decrease performance.

Overriding virtual functions in Python
======================================

Suppose that a C++ class or interface has a virtual function that we'd like to
to override from within Python (we'll focus on the class ``Animal``; ``Dog`` is
given as a specific example of how one would do this with traditional C++
code).

.. code-block:: cpp

    class Animal {
    public:
        virtual ~Animal() { }
        virtual std::string go(int n_times) = 0;
    };

    class Dog : public Animal {
    public:
        std::string go(int n_times) {
            std::string result;
            for (int i=0; i<n_times; ++i)
                result += "woof! ";
            return result;
        }
    };

Let's also suppose that we are given a plain function which calls the
function ``go()`` on an arbitrary ``Animal`` instance.

.. code-block:: cpp

    std::string call_go(Animal *animal) {
        return animal->go(3);
    }

Normally, the binding code for these classes would look as follows:

.. code-block:: cpp

    PYBIND11_PLUGIN(example) {
        py::module m("example", "pybind11 example plugin");

        py::class_<Animal> animal(m, "Animal");
        animal
            .def("go", &Animal::go);

        py::class_<Dog>(m, "Dog", animal)
            .def(py::init<>());

        m.def("call_go", &call_go);

        return m.ptr();
    }

However, these bindings are impossible to extend: ``Animal`` is not
constructible, and we clearly require some kind of "trampoline" that
redirects virtual calls back to Python.

Defining a new type of ``Animal`` from within Python is possible but requires a
helper class that is defined as follows:

.. code-block:: cpp

    class PyAnimal : public Animal {
    public:
        /* Inherit the constructors */
        using Animal::Animal;

        /* Trampoline (need one for each virtual function) */
        std::string go(int n_times) {
            PYBIND11_OVERLOAD_PURE(
                std::string, /* Return type */
                Animal,      /* Parent class */
                go,          /* Name of function */
                n_times      /* Argument(s) */
            );
        }
    };

The macro :func:`PYBIND11_OVERLOAD_PURE` should be used for pure virtual
functions, and :func:`PYBIND11_OVERLOAD` should be used for functions which have
a default implementation. The binding code also needs a few minor adaptations
(highlighted):

.. code-block:: cpp
    :emphasize-lines: 4,6,7

    PYBIND11_PLUGIN(example) {
        py::module m("example", "pybind11 example plugin");

        py::class_<PyAnimal> animal(m, "Animal");
        animal
            .alias<Animal>()
            .def(py::init<>())
            .def("go", &Animal::go);

        py::class_<Dog>(m, "Dog", animal)
            .def(py::init<>());

        m.def("call_go", &call_go);

        return m.ptr();
    }

Importantly, the trampoline helper class is used as the template argument to
:class:`class_`, and a call to :func:`class_::alias` informs the binding
generator that this is merely an alias for the underlying type ``Animal``.
Following this, we are able to define a constructor as usual.

The Python session below shows how to override ``Animal::go`` and invoke it via
a virtual method call.

.. code-block:: cpp

    >>> from example import *
    >>> d = Dog()
    >>> call_go(d)
    u'woof! woof! woof! '
    >>> class Cat(Animal):
    ...     def go(self, n_times):
    ...             return "meow! " * n_times
    ...
    >>> c = Cat()
    >>> call_go(c)
    u'meow! meow! meow! '

.. seealso::

    The file :file:`example/example12.cpp` contains a complete example that
    demonstrates how to override virtual functions using pybind11 in more
    detail.


Global Interpreter Lock (GIL)
=============================

The classes :class:`gil_scoped_release` and :class:`gil_scoped_acquire` can be
used to acquire and release the global interpreter lock in the body of a C++
function call. In this way, long-running C++ code can be parallelized using
multiple Python threads. Taking the previous section as an example, this could
be realized as follows (important changes highlighted):

.. code-block:: cpp
    :emphasize-lines: 8,9,33,34

    class PyAnimal : public Animal {
    public:
        /* Inherit the constructors */
        using Animal::Animal;

        /* Trampoline (need one for each virtual function) */
        std::string go(int n_times) {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERLOAD_PURE(
                std::string, /* Return type */
                Animal,      /* Parent class */
                go,          /* Name of function */
                n_times      /* Argument(s) */
            );
        }
    };

    PYBIND11_PLUGIN(example) {
        py::module m("example", "pybind11 example plugin");

        py::class_<PyAnimal> animal(m, "Animal");
        animal
            .alias<Animal>()
            .def(py::init<>())
            .def("go", &Animal::go);

        py::class_<Dog>(m, "Dog", animal)
            .def(py::init<>());

        m.def("call_go", [](Animal *animal) -> std::string {
            /* Release GIL before calling into (potentially long-running) C++ code */
            py::gil_scoped_release release;
            return call_go(animal);
        });

        return m.ptr();
    }

Passing STL data structures
===========================

When including the additional header file :file:`pybind11/stl.h`, conversions
between ``std::vector<>``, ``std::set<>``, and ``std::map<>`` and the Python
``list``, ``set`` and ``dict`` data structures are automatically enabled. The
types ``std::pair<>`` and ``std::tuple<>`` are already supported out of the box
with just the core :file:`pybind11/pybind11.h` header.

.. note::

    Arbitrary nesting of any of these types is supported.

.. seealso::

    The file :file:`example/example2.cpp` contains a complete example that
    demonstrates how to pass STL data types in more detail.

Binding sequence data types, the slicing protocol, etc.
=======================================================

Please refer to the supplemental example for details.

.. seealso::

    The file :file:`example/example6.cpp` contains a complete example that
    shows how to bind a sequence data type, including length queries
    (``__len__``), iterators (``__iter__``), the slicing protocol and other
    kinds of useful operations.

Return value policies
=====================

Python and C++ use wildly different ways of managing the memory and lifetime of
objects managed by them. This can lead to issues when creating bindings for
functions that return a non-trivial type. Just by looking at the type
information, it is not clear whether Python should take charge of the returned
value and eventually free its resources, or if this is handled on the C++ side.
For this reason, pybind11 provides a several `return value policy` annotations
that can be passed to the :func:`module::def` and :func:`class_::def`
functions. The default policy is :enum:`return_value_policy::automatic`.


+--------------------------------------------------+---------------------------------------------------------------------------+
| Return value policy                              | Description                                                               |
+==================================================+===========================================================================+
| :enum:`return_value_policy::automatic`           | Automatic: copy objects returned as values and take ownership of          |
|                                                  | objects returned as pointers                                              |
+--------------------------------------------------+---------------------------------------------------------------------------+
| :enum:`return_value_policy::copy`                | Create a new copy of the returned object, which will be owned by Python   |
+--------------------------------------------------+---------------------------------------------------------------------------+
| :enum:`return_value_policy::take_ownership`      | Reference the existing object and take ownership. Python will call        |
|                                                  | the destructor and delete operator when the reference count reaches zero  |
+--------------------------------------------------+---------------------------------------------------------------------------+
| :enum:`return_value_policy::reference`           | Reference the object, but do not take ownership and defer responsibility  |
|                                                  | for deleting it to C++ (dangerous when C++ code at some point decides to  |
|                                                  | delete it while Python still has a nonzero reference count)               |
+--------------------------------------------------+---------------------------------------------------------------------------+
| :enum:`return_value_policy::reference_internal`  | Reference the object, but do not take ownership. The object is considered |
|                                                  | be owned by the C++ instance whose method or property returned it. The    |
|                                                  | Python object will increase the reference count of this 'parent' by 1     |
|                                                  | to ensure that it won't be deallocated while Python is using the 'child'  |
+--------------------------------------------------+---------------------------------------------------------------------------+

.. warning::

    Code with invalid call policies might access unitialized memory and free
    data structures multiple times, which can lead to hard-to-debug
    non-determinism and segmentation faults, hence it is worth spending the
    time to understand all the different options above.

See below for an example that uses the
:enum:`return_value_policy::reference_internal` policy.

.. code-block:: cpp

    class Example {
    public:
        Internal &get_internal() { return internal; }
    private:
        Internal internal;
    };

    PYBIND11_PLUGIN(example) {
        py::module m("example", "pybind11 example plugin");

        py::class_<Example>(m, "Example")
            .def(py::init<>())
            .def("get_internal", &Example::get_internal, "Return the internal data", py::return_value_policy::reference_internal)

        return m.ptr();
    }


Additional call policies
========================

In addition to the above return value policies, further `call policies` can be
specified to indicate dependencies between parameters. There is currently just
one policy named ``keep_alive<Nurse, Patient>``, which indicates that the
argument with index ``Patient`` should be kept alive at least until the
argument with index ``Nurse`` is freed by the garbage collector; argument
indices start at one, while zero refers to the return value. Arbitrarily many
call policies can be specified.

For instance, binding code for a a list append operation that ties the lifetime
of the newly added element to the underlying container might be declared as
follows:

.. code-block:: cpp

    py::class_<List>(m, "List")
        .def("append", &List::append, py::keep_alive<1, 2>());

.. note::

    ``keep_alive`` is analogous to the ``with_custodian_and_ward`` (if Nurse,
    Patient != 0) and ``with_custodian_and_ward_postcall`` (if Nurse/Patient ==
    0) policies from Boost.Python.

Implicit type conversions
=========================

Suppose that instances of two types ``A`` and ``B`` are used in a project, and
that an ``A`` can easily be converted into a an instance of type ``B`` (examples of this
could be a fixed and an arbitrary precision number type).

.. code-block:: cpp

    py::class_<A>(m, "A")
        /// ... members ...

    py::class_<B>(m, "B")
        .def(py::init<A>())
        /// ... members ...

    m.def("func",
        [](const B &) { /* .... */ }
    );

To invoke the function ``func`` using a variable ``a`` containing an ``A``
instance, we'd have to write ``func(B(a))`` in Python. On the other hand, C++
will automatically apply an implicit type conversion, which makes it possible
to directly write ``func(a)``.

In this situation (i.e. where ``B`` has a constructor that converts from
``A``), the following statement enables similar implicit conversions on the
Python side:

.. code-block:: cpp

    py::implicitly_convertible<A, B>();

Smart pointers
==============

The binding generator for classes (:class:`class_`) takes an optional second
template type, which denotes a special *holder* type that is used to manage
references to the object. When wrapping a type named ``Type``, the default
value of this template parameter is ``std::unique_ptr<Type>``, which means that
the object is deallocated when Python's reference count goes to zero.

It is possible to switch to other types of reference counting wrappers or smart
pointers, which is useful in codebases that rely on them. For instance, the
following snippet causes ``std::shared_ptr`` to be used instead.

.. code-block:: cpp

    py::class_<Example, std::shared_ptr<Example> /* <- holder type */> obj(m, "Example");

Note that any particular class can only be associated with a single holder type.

To enable transparent conversions for functions that take shared pointers as an
argument or that return them, a macro invocation similar to the following must
be declared at the top level before any binding code:

.. code-block:: cpp

    PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

.. note::

    The first argument of :func:`PYBIND11_DECLARE_HOLDER_TYPE` should be a
    placeholder name that is used as a template parameter of the second
    argument. Thus, feel free to use any identifier, but use it consistently on
    both sides; also, don't use the name of a type that already exists in your
    codebase.

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

    PYBIND11_PLUGIN(example) {
        py::module m("example");

        py::class_<Child, std::shared_ptr<Child>>(m, "Child");

        py::class_<Parent, std::shared_ptr<Parent>>(m, "Parent")
           .def(py::init<>())
           .def("get_child", &Parent::get_child);

        return m.ptr();
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

    The file :file:`example/example8.cpp` contains a complete example that
    demonstrates how to work with custom reference-counting holder types in
    more detail.

.. _custom_constructors:

Custom constructors
===================

The syntax for binding constructors was previously introduced, but it only
works when a constructor with the given parameters actually exists on the C++
side. To extend this to more general cases, let's take a look at what actually
happens under the hood: the following statement

.. code-block:: cpp

    py::class_<Example>(m, "Example")
        .def(py::init<int>());

is short hand notation for

.. code-block:: cpp

    py::class_<Example>(m, "Example")
        .def("__init__",
            [](Example &instance, int arg) {
                new (&instance) Example(arg);
            }
        );

In other words, :func:`init` creates an anonymous function that invokes an
in-place constructor. Memory allocation etc. is already take care of beforehand
within pybind11.

Catching and throwing exceptions
================================

When C++ code invoked from Python throws an ``std::exception``, it is
automatically converted into a Python ``Exception``. pybind11 defines multiple
special exception classes that will map to different types of Python
exceptions:

+----------------------------+------------------------------+
|  C++ exception type        |  Python exception type       |
+============================+==============================+
| :class:`std::exception`    | ``Exception``                |
+----------------------------+------------------------------+
| :class:`stop_iteration`    | ``StopIteration`` (used to   |
|                            | implement custom iterators)  |
+----------------------------+------------------------------+
| :class:`index_error`       | ``IndexError`` (used to      |
|                            | indicate out of bounds       |
|                            | accesses in ``__getitem__``, |
|                            | ``__setitem__``, etc.)       |
+----------------------------+------------------------------+
| :class:`error_already_set` | Indicates that the Python    |
|                            | exception flag has already   |
|                            | been initialized.            |
+----------------------------+------------------------------+

When a Python function invoked from C++ throws an exception, it is converted
into a C++ exception of type :class:`error_already_set` whose string payload
contains a textual summary.

There is also a special exception :class:`cast_error` that is thrown by
:func:`handle::call` when the input arguments cannot be converted to Python
objects.

Buffer protocol
===============

Python supports an extremely general and convenient approach for exchanging
data between plugin libraries. Types can expose a buffer view which provides
fast direct access to the raw internal representation. Suppose we want to bind
the following simplistic Matrix class:

.. code-block:: cpp

    class Matrix {
    public:
        Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols) {
            m_data = new float[rows*cols];
        }
        float *data() { return m_data; }
        size_t rows() const { return m_rows; }
        size_t cols() const { return m_cols; }
    private:
        size_t m_rows, m_cols;
        float *m_data;
    };

The following binding code exposes the ``Matrix`` contents as a buffer object,
making it possible to cast Matrixes into NumPy arrays. It is even possible to
completely avoid copy operations with Python expressions like
``np.array(matrix_instance, copy = False)``.

.. code-block:: cpp

    py::class_<Matrix>(m, "Matrix")
       .def_buffer([](Matrix &m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),                              /* Pointer to buffer */
                sizeof(float),                         /* Size of one scalar */
                py::format_descriptor<float>::value(), /* Python struct-style format descriptor */
                2,                                     /* Number of dimensions */
                { m.rows(), m.cols() },                /* Buffer dimensions */
                { sizeof(float) * m.rows(),            /* Strides (in bytes) for each index */
                  sizeof(float) }
            );
        });

The snippet above binds a lambda function, which can create ``py::buffer_info``
description records on demand describing a given matrix. The contents of
``py::buffer_info`` mirror the Python buffer protocol specification.

.. code-block:: cpp

    struct buffer_info {
        void *ptr;
        size_t itemsize;
        std::string format;
        int ndim;
        std::vector<size_t> shape;
        std::vector<size_t> strides;
    };

To create a C++ function that can take a Python buffer object as an argument,
simply use the type ``py::buffer`` as one of its arguments. Buffers can exist
in a great variety of configurations, hence some safety checks are usually
necessary in the function body. Below, you can see an basic example on how to
define a custom constructor for the Eigen double precision matrix
(``Eigen::MatrixXd``) type, which supports initialization from compatible
buffer
objects (e.g. a NumPy matrix).

.. code-block:: cpp

    py::class_<Eigen::MatrixXd>(m, "MatrixXd")
        .def("__init__", [](Eigen::MatrixXd &m, py::buffer b) {
            /* Request a buffer descriptor from Python */
            py::buffer_info info = b.request();

            /* Some sanity checks ... */
            if (info.format != py::format_descriptor<double>::value())
                throw std::runtime_error("Incompatible format: expected a double array!");

            if (info.ndim != 2)
                throw std::runtime_error("Incompatible buffer dimension!");

            if (info.strides[0] == sizeof(double)) {
                /* Buffer has the right layout -- directly copy. */
                new (&m) Eigen::MatrixXd(info.shape[0], info.shape[1]);
                memcpy(m.data(), info.ptr, sizeof(double) * m.size());
            } else {
                /* Oops -- the buffer is transposed */
                new (&m) Eigen::MatrixXd(info.shape[1], info.shape[0]);
                memcpy(m.data(), info.ptr, sizeof(double) * m.size());
                m.transposeInPlace();
            }
        });

.. seealso::

    The file :file:`example/example7.cpp` contains a complete example that
    demonstrates using the buffer protocol with pybind11 in more detail.

NumPy support
=============

By exchanging ``py::buffer`` with ``py::array`` in the above snippet, we can
restrict the function so that it only accepts NumPy arrays (rather than any
type of Python object satisfying the buffer object protocol).

In many situations, we want to define a function which only accepts a NumPy
array of a certain data type. This is possible via the ``py::array_t<T>``
template. For instance, the following function requires the argument to be a
dense array of doubles in C-style ordering.

.. code-block:: cpp

    void f(py::array_t<double> array);

When it is invoked with a different type (e.g. an integer), the binding code
will attempt to cast the input into a NumPy array of the requested type.
Note that this feature requires the ``pybind11/numpy.h`` header to be included.

Vectorizing functions
=====================

Suppose we want to bind a function with the following signature to Python so
that it can process arbitrary NumPy array arguments (vectors, matrices, general
N-D arrays) in addition to its normal arguments:

.. code-block:: cpp

    double my_func(int x, float y, double z);

After including the ``pybind11/numpy.h`` header, this is extremely simple:

.. code-block:: cpp

    m.def("vectorized_func", py::vectorize(my_func));

Invoking the function like below causes 4 calls to be made to ``my_func`` with
each of the the array elements. The result is returned as a NumPy array of type
``numpy.dtype.float64``.

.. code-block:: python

    >>> x = np.array([[1, 3],[5, 7]])
    >>> y = np.array([[2, 4],[6, 8]])
    >>> z = 3
    >>> result = vectorized_func(x, y, z)

The scalar argument ``z`` is transparently replicated 4 times.  The input
arrays ``x`` and ``y`` are automatically converted into the right types (they
are of type  ``numpy.dtype.int64`` but need to be ``numpy.dtype.int32`` and
``numpy.dtype.float32``, respectively)

Sometimes we might want to explitly exclude an argument from the vectorization
because it makes little sense to wrap it in a NumPy array. For instance,
suppose the function signature was

.. code-block:: cpp

    double my_func(int x, float y, my_custom_type *z);

This can be done with a stateful Lambda closure:

.. code-block:: cpp

    // Vectorize a lambda function with a capture object (e.g. to exclude some arguments from the vectorization)
    m.def("vectorized_func",
        [](py::array_t<int> x, py::array_t<float> y, my_custom_type *z) {
            auto stateful_closure = [z](int x, float y) { return my_func(x, y, z); };
            return py::vectorize(stateful_closure)(x, y);
        }
    );

.. seealso::

    The file :file:`example/example10.cpp` contains a complete example that
    demonstrates using :func:`vectorize` in more detail.

Functions taking Python objects as arguments
============================================

pybind11 exposes all major Python types using thin C++ wrapper classes. These
wrapper classes can also be used as parameters of functions in bindings, which
makes it possible to directly work with native Python types on the C++ side.
For instance, the following statement iterates over a Python ``dict``:

.. code-block:: cpp

    void print_dict(py::dict dict) {
        /* Easily interact with Python types */
        for (auto item : dict)
            std::cout << "key=" << item.first << ", "
                      << "value=" << item.second << std::endl;
    }

Available types include :class:`handle`, :class:`object`, :class:`bool_`,
:class:`int_`, :class:`float_`, :class:`str`, :class:`bytes`, :class:`tuple`,
:class:`list`, :class:`dict`, :class:`slice`, :class:`capsule`,
:class:`function`, :class:`buffer`, :class:`array`, and :class:`array_t`.

In this kind of mixed code, it is often necessary to convert arbitrary C++
types to Python, which can be done using :func:`cast`:

.. code-block:: cpp

    MyClass *cls = ..;
    py::object obj = py::cast(cls);

The reverse direction uses the following syntax:

.. code-block:: cpp

    py::object obj = ...;
    MyClass *cls = obj.cast<MyClass *>();

When conversion fails, both directions throw the exception :class:`cast_error`.

.. seealso::

    The file :file:`example/example2.cpp` contains a complete example that
    demonstrates passing native Python types in more detail.

Default arguments revisited
===========================

The section on :ref:`default_args` previously discussed basic usage of default
arguments using pybind11. One noteworthy aspect of their implementation is that
default arguments are converted to Python objects right at declaration time.
Consider the following example:

.. code-block:: cpp

    py::class_<MyClass>("MyClass")
        .def("myFunction", py::arg("arg") = SomeType(123));

In this case, pybind11 must already be set up to deal with values of the type
``SomeType`` (via a prior instantiation of ``py::class_<SomeType>``), or an
exception will be thrown.

Another aspect worth highlighting is that the "preview" of the default argument
in the function signature is generated using the object's ``__repr__`` method.
If not available, the signature may not be very helpful, e.g.:

.. code-block:: python

    FUNCTIONS
    ...
    |  myFunction(...)
    |      Signature : (MyClass, arg : SomeType = <SomeType object at 0x101b7b080>) -> None
    ...

The first way of addressing this is by defining ``SomeType.__repr__``.
Alternatively, it is possible to specify the human-readable preview of the
default argument manually using the ``arg_t`` notation:

.. code-block:: cpp

    py::class_<MyClass>("MyClass")
        .def("myFunction", py::arg_t<SomeType>("arg", SomeType(123), "SomeType(123)"));

Partitioning code over multiple extension modules
=================================================

It's straightforward to split binding code over multiple extension modules and
reference types declared elsewhere. Everything "just" works without any special
precautions. One exception to this rule occurs when wanting to extend a type declared
in another extension module. Recall the basic example from Section
:ref:`inheritance`.

.. code-block:: cpp

    py::class_<Pet> pet(m, "Pet");
    pet.def(py::init<const std::string &>())
       .def_readwrite("name", &Pet::name);

    py::class_<Dog>(m, "Dog", pet /* <- specify parent */)
        .def(py::init<const std::string &>())
        .def("bark", &Dog::bark);

Suppose now that ``Pet`` bindings are defined in a module named ``basic``,
whereas the ``Dog`` bindings are defined somewhere else. The challenge is of
course that the variable ``pet`` is not available anymore though it is needed
to indicate the inheritance relationship to the constructor of ``class_<Dog>``.
However, it can be acquired as follows:

.. code-block:: cpp

    py::object pet = (py::object) py::module::import("basic").attr("Pet");

    py::class_<Dog>(m, "Dog", pet)
        .def(py::init<const std::string &>())
        .def("bark", &Dog::bark);

