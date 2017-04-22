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
