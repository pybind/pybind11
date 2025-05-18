.. _embedding:

Embedding the interpreter
#########################

While pybind11 is mainly focused on extending Python using C++, it's also
possible to do the reverse: embed the Python interpreter into a C++ program.
All of the other documentation pages still apply here, so refer to them for
general pybind11 usage. This section will cover a few extra things required
for embedding.

Getting started
===============

A basic executable with an embedded interpreter can be created with just a few
lines of CMake and the ``pybind11::embed`` target, as shown below. For more
information, see :doc:`/compiling`.

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.15...4.0)
    project(example)

    find_package(pybind11 REQUIRED)  # or `add_subdirectory(pybind11)`

    add_executable(example main.cpp)
    target_link_libraries(example PRIVATE pybind11::embed)

The essential structure of the ``main.cpp`` file looks like this:

.. code-block:: cpp

    #include <pybind11/embed.h> // everything needed for embedding
    namespace py = pybind11;

    int main() {
        py::scoped_interpreter guard{}; // start the interpreter and keep it alive

        py::print("Hello, World!"); // use the Python API
    }

The interpreter must be initialized before using any Python API, which includes
all the functions and classes in pybind11. The RAII guard class ``scoped_interpreter``
takes care of the interpreter lifetime. After the guard is destroyed, the interpreter
shuts down and clears its memory. No Python functions can be called after this.

Executing Python code
=====================

There are a few different ways to run Python code. One option is to use ``eval``,
``exec`` or ``eval_file``, as explained in :ref:`eval`. Here is a quick example in
the context of an executable with an embedded interpreter:

.. code-block:: cpp

    #include <pybind11/embed.h>
    namespace py = pybind11;

    int main() {
        py::scoped_interpreter guard{};

        py::exec(R"(
            kwargs = dict(name="World", number=42)
            message = "Hello, {name}! The answer is {number}".format(**kwargs)
            print(message)
        )");
    }

Alternatively, similar results can be achieved using pybind11's API (see
:doc:`/advanced/pycpp/index` for more details).

.. code-block:: cpp

    #include <pybind11/embed.h>
    namespace py = pybind11;
    using namespace py::literals;

    int main() {
        py::scoped_interpreter guard{};

        auto kwargs = py::dict("name"_a="World", "number"_a=42);
        auto message = "Hello, {name}! The answer is {number}"_s.format(**kwargs);
        py::print(message);
    }

The two approaches can also be combined:

.. code-block:: cpp

    #include <pybind11/embed.h>
    #include <iostream>

    namespace py = pybind11;
    using namespace py::literals;

    int main() {
        py::scoped_interpreter guard{};

        auto locals = py::dict("name"_a="World", "number"_a=42);
        py::exec(R"(
            message = "Hello, {name}! The answer is {number}".format(**locals())
        )", py::globals(), locals);

        auto message = locals["message"].cast<std::string>();
        std::cout << message;
    }

Importing modules
=================

Python modules can be imported using ``module_::import()``:

.. code-block:: cpp

    py::module_ sys = py::module_::import("sys");
    py::print(sys.attr("path"));

For convenience, the current working directory is included in ``sys.path`` when
embedding the interpreter. This makes it easy to import local Python files:

.. code-block:: python

    """calc.py located in the working directory"""


    def add(i, j):
        return i + j


.. code-block:: cpp

    py::module_ calc = py::module_::import("calc");
    py::object result = calc.attr("add")(1, 2);
    int n = result.cast<int>();
    assert(n == 3);

Modules can be reloaded using ``module_::reload()`` if the source is modified e.g.
by an external process. This can be useful in scenarios where the application
imports a user defined data processing script which needs to be updated after
changes by the user. Note that this function does not reload modules recursively.

.. _embedding_modules:

Adding embedded modules
=======================

Embedded binary modules can be added using the ``PYBIND11_EMBEDDED_MODULE`` macro.
Note that the definition must be placed at global scope. They can be imported
like any other module.

.. code-block:: cpp

    #include <pybind11/embed.h>
    namespace py = pybind11;

    PYBIND11_EMBEDDED_MODULE(fast_calc, m) {
        // `m` is a `py::module_` which is used to bind functions and classes
        m.def("add", [](int i, int j) {
            return i + j;
        });
    }

    int main() {
        py::scoped_interpreter guard{};

        auto fast_calc = py::module_::import("fast_calc");
        auto result = fast_calc.attr("add")(1, 2).cast<int>();
        assert(result == 3);
    }

Unlike extension modules where only a single binary module can be created, on
the embedded side an unlimited number of modules can be added using multiple
``PYBIND11_EMBEDDED_MODULE`` definitions (as long as they have unique names).

These modules are added to Python's list of builtins, so they can also be
imported in pure Python files loaded by the interpreter. Everything interacts
naturally:

.. code-block:: python

    """py_module.py located in the working directory"""
    import cpp_module

    a = cpp_module.a
    b = a + 1


.. code-block:: cpp

    #include <pybind11/embed.h>
    namespace py = pybind11;

    PYBIND11_EMBEDDED_MODULE(cpp_module, m) {
        m.attr("a") = 1;
    }

    int main() {
        py::scoped_interpreter guard{};

        auto py_module = py::module_::import("py_module");

        auto locals = py::dict("fmt"_a="{} + {} = {}", **py_module.attr("__dict__"));
        assert(locals["a"].cast<int>() == 1);
        assert(locals["b"].cast<int>() == 2);

        py::exec(R"(
            c = a + b
            message = fmt.format(a, b, c)
        )", py::globals(), locals);

        assert(locals["c"].cast<int>() == 3);
        assert(locals["message"].cast<std::string>() == "1 + 2 = 3");
    }

``PYBIND11_EMBEDDED_MODULE`` also accepts
:func:`py::mod_gil_not_used()`,
:func:`py::multiple_interpreters::per_interpreter_gil()`, and
:func:`py::multiple_interpreters::shared_gil()` tags just like ``PYBIND11_MODULE``.
See :ref:`misc_subinterp` and :ref:`misc_free_threading` for more information.

Interpreter lifetime
====================

The Python interpreter shuts down when ``scoped_interpreter`` is destroyed. After
this, creating a new instance will restart the interpreter. Alternatively, the
``initialize_interpreter`` / ``finalize_interpreter`` pair of functions can be used
to directly set the state at any time.

Modules created with pybind11 can be safely re-initialized after the interpreter
has been restarted. However, this may not apply to third-party extension modules.
The issue is that Python itself cannot completely unload extension modules and
there are several caveats with regard to interpreter restarting. In short, not
all memory may be freed, either due to Python reference cycles or user-created
global data. All the details can be found in the CPython documentation.

.. warning::

    Creating two concurrent ``scoped_interpreter`` guards is a fatal error. So is
    calling ``initialize_interpreter`` for a second time after the interpreter
    has already been initialized. Use :class:`scoped_subinterpreter` to create
    a sub-interpreter.  See :ref:`subinterp` for important details on sub-interpreters.

    Do not use the raw CPython API functions ``Py_Initialize`` and
    ``Py_Finalize`` as these do not properly handle the lifetime of
    pybind11's internal data.


.. _subinterp:

Sub-interpreter support
=======================

A sub-interpreter is a separate interpreter instance which provides a
separate, isolated interpreter environment within the same process as the main
interpreter.  Sub-interpreters are created and managed with a separate API from
the main interpreter. Beginning in Python 3.12, sub-interpreters each have
their own Global Interpreter Lock (GIL), which means that running a
sub-interpreter in a separate thread from the main interpreter can achieve true
concurrency.

Managing multiple threads and the lifetimes of multiple interpreters and their
GILs can be challenging.  Proceed with caution (and lots of testing)!

The main interpreter must be initialized before creating a sub-interpreter, and
the main interpreter must outlive all sub-interpreters. Sub-interpreters are
managed through a different API than the main interpreter.

The sub-interpreter API can be found in ``pybind11/subinterpreter.h``.

The :class:`subinterpreter` class manages the lifetime of sub-interpreters.
Instances are movable, but not copyable. Default constructing this class does
*not* create a sub-interpreter (it creates an empty holder).  To create a
sub-interpreter, call :func:`subinterpreter::create()`.

.. warning::

    Sub-interpreter creation acquires (and subsequently releases) the main
    interpreter GIL. If another thread holds the main GIL, the function will
    block until the main GIL can be acquired.

    Sub-interpreter destruction temporarily activates the sub-interpreter. The
    sub-interpreter must not be active (on any threads) at the time the
    :class:`subinterpreter` destructor is called.

    Both actions will re-acquire any interpreter's GIL that was held prior to
    the call before returning (or return to no active interpreter if none was
    active at the time of the call).

Once a sub-interpreter is created, you can "activate" it on a thread (and
acquire it's GIL) by creating a :class:`subinterpreter_scoped_activate`
instance and passing it the sub-intepreter to be activated.  The function
will acquire the sub-interpreter's GIL and make the sub-interpreter the
current active interpreter on the current thread for the lifetime of the
instance. When the :class:`subinterpreter_scoped_activate` instance goes out
of scope, the sub-interpreter GIL is released and the prior interpreter that
was active on the thread (if any) is reactivated and it's GIL is re-acquired.

The :func:`subinterpreter::activate_main()` function activates the main
interpreter, acquiring it's GIL, and returns a
:class:`subinterpreter_scoped_activate` instance which will automatically
deactivate the main interpreter and release it's GIL when it goes out of
scope, just as :class:`subinterpreter_scoped_activate` also does for
sub-interpreters.

:class:`gil_scoped_release` and :class:`gil_scoped_acquire` can be used to
manage the GIL of a sub-interpreter just as they do for the main interpreter.
They both manage the GIL of the currently active interpreter, without the
programmer having to do anything special or different. There is one important 
caveat:

.. note::

    When no interpreter is active through a
    :class:`subinterpreter_scoped_activate` instance (such as on a new thread),
    :class:`gil_scoped_acquire` will acquire the **main** GIL and
    activate the **main** interpreter.

Each sub-interpreter will import a separate copy of each ``PYBIND11_EMBEDDED_MODULE``
when those modules specify a ``multiple_interpreters`` tag. If a module does not
specify a ``multiple_interpreters`` tag, then Python will report an ``ImportError`` 
if it is imported in a sub-interpreter.

Here is an example showing how to create and activate sub-interpreters:

.. code-block:: cpp

    #include <iostream>
    #include <pybind11/embed.h>
    #include <pybind11/subinterpreter.h>

    namespace py = pybind11;

    PYBIND11_EMBEDDED_MODULE(printer, m, py::multiple_interpreters::per_interpreter_gil()) {
        m.def("which", [](const std::string& when) {
            std::cout << when << "; Current Interpreter is "
                    << PyInterpreterState_GetID(PyInterpreterState_Get())
                    << std::endl;
        });
    }

    int main() {
        py::scoped_interpreter main_int{};

        py::module_::import("printer").attr("which")("First init");

        {
            py::subinterpreter sub = py::subinterpreter::create();

            py::module_::import("printer").attr("which")("Created sub");

            {
                py::subinterpreter_scoped_activate ssa(sub);
                py::module_::import("printer").attr("which")("Activated sub");
            }

            py::module_::import("printer").attr("which")("Deactivated sub");

            {
                py::gil_scoped_release nogil;
                {
                    py::subinterpreter_scoped_activate ssa(sub);
                    {
                        auto main_sa = py::subinterpreter::main_scoped_activate();
                        py::module_::import("printer").attr("which")("Main within sub");
                    }
                    py::module_::import("printer").attr("which")("After Main, still within sub");
                }
            }
        }

        py::module_::import("printer").attr("which")("At end");

        return 0;
    }

Expected output:

.. code-block:: text

    First init; Current Interpreter is 0
    Created sub; Current Interpreter is 0
    Activated sub; Current Interpreter is 1
    Deactivated sub; Current Interpreter is 0
    Main within sub; Current Interpreter is 0
    After Main, still within sub; Current Interpreter is 1
    At end; Current Interpreter is 0

pybind11 also has a :class:`scoped_subinterpreter` class, which creates and
activates a sub-interpreter when it is constructed, and deactivates and deletes
it when it goes out of scope.

Best Practices for sub-interpreter safety:

- Never share Python objects across different interpreters.

- Avoid global/static state whenever possible. Instead, keep state within each interpreter,
  such as within the interpreter state dict, which can be accessed via
  ``subinterpreter::current().state_dict()``, or within instance members and tied to
  Python objects.

- Avoid trying to "cache" Python objects in C++ variables across function calls (this is an easy
  way to accidentally introduce sub-interpreter bugs). In the code example above, note that we
  did not save the result of :func:`module_::import`, in order to avoid accidentally using the
  resulting Python object when the wrong interpreter was active.

- While sub-interpreters each have their own GIL, there can now be multiple independent GILs in one
  program, so your code needs to consider thread safety of within the C++ code, and the possibility
  of deadlocks caused by multiple GILs and/or the interactions of the GIL(s) and C++'s own locking.
