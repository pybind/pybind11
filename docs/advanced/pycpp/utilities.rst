Utilities
#########

Using Python's print function in C++
====================================

The usual way to write output in C++ is using ``std::cout`` while in Python one
would use ``print``. Since these methods use different buffers, mixing them can
lead to output order issues. To resolve this, pybind11 modules can use the
:func:`py::print` function which writes to Python's ``sys.stdout`` for consistency.

Python's ``print`` function is replicated in the C++ API including optional
keyword arguments ``sep``, ``end``, ``file``, ``flush``. Everything works as
expected in Python:

.. code-block:: cpp

    py::print(1, 2.0, "three"); // 1 2.0 three
    py::print(1, 2.0, "three", "sep"_a="-"); // 1-2.0-three

    auto args = py::make_tuple("unpacked", true);
    py::print("->", *args, "end"_a="<-"); // -> unpacked True <-

Executing code in a Python `with` statement
===========================================

Where in C++, the scope of local variables and their destructor is used to manage
resources in the RAII idiom, Python has a `with` statement and context managers
for such situations. At the start and end of the `with` statement, the context
manager's `__enter__` and `__exit__` methods are called, respectively. To more
easily use context managers in a C++ context, pybind11 provides a utility function
``py::with``, that matches the semantics of a Python `with`-statement (see
`PEP 343 <https://www.python.org/dev/peps/pep-0343/>`_):

.. code-block:: cpp

	auto io = py::module::import("io");
	py::with(io.attr("open")("tmp.out", "w"), [](py::object &&f) {
        for (int i = 0; i < 10; ++i) {
            f.attr("write")(i);
            f.attr("write")("\n");
        }
	});

This code snippet corresponds to the following in Python:

.. code-block:: python

    import io
    with io.open("tmp.out", "w") as f:
        for i in range(10):
            f.write(i)
            f.write("\n")

The `py::object` parameter of the lambda function can be omitted if the object resulting
from the context manager (i.e., the `as VAR` part in the `with` statement) is not of use.

Optionally, an extra `py::with_exception_policy` argument can be passed to `py::with`.
If the value `py::with_exception_policy::translate` is selected, pybind11 will try to
translate any C++ exception inside the `with` statement and pass the Python exception
as argument into the `__exit__` method of the context manager (cfr. PEP 343). If
`py::with_exception_policy::translate` is passed and an exception gets thrown, pybind11
will not try to translate it, `__exit__` will be called as if no exception was thrown,
and the original exception will be cascaded down to the caller.

.. _ostream_redirect:

Capturing standard output from ostream
======================================

Often, a library will use the streams ``std::cout`` and ``std::cerr`` to print,
but this does not play well with Python's standard ``sys.stdout`` and ``sys.stderr``
redirection. Replacing a library's printing with `py::print <print>` may not
be feasible. This can be fixed using a guard around the library function that
redirects output to the corresponding Python streams:

.. code-block:: cpp

    #include <pybind11/iostream.h>

    ...

    // Add a scoped redirect for your noisy code
    m.def("noisy_func", []() {
        py::scoped_ostream_redirect stream(
            std::cout,                               // std::ostream&
            py::module::import("sys").attr("stdout") // Python output
        );
        call_noisy_func();
    });

This method respects flushes on the output streams and will flush if needed
when the scoped guard is destroyed. This allows the output to be redirected in
real time, such as to a Jupyter notebook. The two arguments, the C++ stream and
the Python output, are optional, and default to standard output if not given. An
extra type, `py::scoped_estream_redirect <scoped_estream_redirect>`, is identical
except for defaulting to ``std::cerr`` and ``sys.stderr``; this can be useful with
`py::call_guard`, which allows multiple items, but uses the default constructor:

.. code-block:: py

    // Alternative: Call single function using call guard
    m.def("noisy_func", &call_noisy_function,
          py::call_guard<py::scoped_ostream_redirect,
                         py::scoped_estream_redirect>());

The redirection can also be done in Python with the addition of a context
manager, using the `py::add_ostream_redirect() <add_ostream_redirect>` function:

.. code-block:: cpp

    py::add_ostream_redirect(m, "ostream_redirect");

The name in Python defaults to ``ostream_redirect`` if no name is passed.  This
creates the following context manager in Python:

.. code-block:: python

    with ostream_redirect(stdout=True, stderr=True):
        noisy_function()

It defaults to redirecting both streams, though you can use the keyword
arguments to disable one of the streams if needed.

.. note::

    The above methods will not redirect C-level output to file descriptors, such
    as ``fprintf``. For those cases, you'll need to redirect the file
    descriptors either directly in C or with Python's ``os.dup2`` function
    in an operating-system dependent way.

.. _eval:

Evaluating Python expressions from strings and files
====================================================

pybind11 provides the `eval`, `exec` and `eval_file` functions to evaluate
Python expressions and statements. The following example illustrates how they
can be used.

.. code-block:: cpp

    // At beginning of file
    #include <pybind11/eval.h>

    ...

    // Evaluate in scope of main module
    py::object scope = py::module::import("__main__").attr("__dict__");

    // Evaluate an isolated expression
    int result = py::eval("my_variable + 10", scope).cast<int>();

    // Evaluate a sequence of statements
    py::exec(
        "print('Hello')\n"
        "print('world!');",
        scope);

    // Evaluate the statements in an separate Python file on disk
    py::eval_file("script.py", scope);

C++11 raw string literals are also supported and quite handy for this purpose.
The only requirement is that the first statement must be on a new line following
the raw string delimiter ``R"(``, ensuring all lines have common leading indent:

.. code-block:: cpp

    py::exec(R"(
        x = get_answer()
        if x == 42:
            print('Hello World!')
        else:
            print('Bye!')
        )", scope
    );

.. note::

    `eval` and `eval_file` accept a template parameter that describes how the
    string/file should be interpreted. Possible choices include ``eval_expr``
    (isolated expression), ``eval_single_statement`` (a single statement, return
    value is always ``none``), and ``eval_statements`` (sequence of statements,
    return value is always ``none``). `eval` defaults to  ``eval_expr``,
    `eval_file` defaults to ``eval_statements`` and `exec` is just a shortcut
    for ``eval<eval_statements>``.
