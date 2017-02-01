Strings, bytes and Unicode conversions
######################################

.. note::

    This section discusses string handling in terms of Python 3 strings. For Python 2.7, replace all occurrences of ``str`` with ``unicode`` and ``bytes`` with ``str``.

Passing Python strings to C++
=============================

When a Python ``str`` is passed from Python to a C++ function that accepts ``std::string`` or ``char *`` as arguments, pybind11 will encode the Python string to UTF-8. All Python ``str`` can be encoded in UTF-8, so this operation does not fail.

The C++ language is encoding agnostic. It is the responsibility of the programmer to track encodings. It's often easiest to simply `use UTF-8 everywhere <http://utf8everywhere.org/>`_.

.. code-block:: c++

    m.def("utf8_test",
        [](const std::string &s) {
            cout << "utf-8 is icing on the cake.\n";
            cout << s;
        }
    );
    m.def("utf8_charptr",
        [](const char *s) {
            cout << "My favorite food is\n";
            cout << s;
        }
    );

.. code-block:: python

    >>> utf8_test('🎂')
    I am going to output utf-8.
    🎂

    >>> utf8_charptr('🍕')
    My favorite food is
    🍕

.. note::

    Some terminal emulators do not support UTF-8 or emoji fonts and may not display the example above correctly.

The results are the same whether the C++ function accepts arguments by value or reference, and whether or not ``const`` is used.

Passing bytes to C++
--------------------

A Python ``bytes`` object will be passed to C++ without conversion.


Returning C++ strings to Python
===============================

When a C++ function returns a ``std::string`` or ``char*`` to a Python caller, **pybind11 will assume that the string is valid UTF-8** and will decode it to a native Python ``str``, using the same API as Python uses to perform ``bytes.decode('utf-8')``. If this implicit conversion fails, pybind11 will raise a ``UnicodeDecodeError``. 

.. code-block:: c++

    m.def("std_string_return",
        []() {
            return std::string("This string needs to be UTF-8 encoded");
        }
    );

.. code-block:: python

    >>> isinstance(example.std_string_return(), str)
    True


Because UTF-8 is inclusive of pure ASCII, there is never any issue with returning a pure ASCII string to Python. If there is any possibility that the string is not pure ASCII, it is necessary to ensure the encoding is valid UTF-8.

.. warning::

    Implicit conversion assumes that a returned ``char *`` is null-terminated. If there is no null terminator a buffer overrun will occur.


Explicit conversions
--------------------

If some C++ code constructs a ``std::string`` that is not a UTF-8 string, one can perform a explicit conversion and return a ``py::str`` object. Explicit conversion has the same overhead as implicit conversion.

.. code-block:: c++

    // This wraps a C++ function that returns a Latin-1 encoded string.
    // It uses the Python C API to convert this to Unicode.
    m.def("str_output",
        []() {
            std::string s = "Send your r\xe9sum\xe9 to Alice in HR"; // Latin-1
            py::str py_s = PyUnicode_DecodeLatin1(s.data(), s.length());
            return py_s;
        }
    );

.. code-block:: python

    >>> str_output()
    'Send your résumé to Alice in HR'

The `Python C API <https://docs.python.org/3/c-api/unicode.html#built-in-codecs>`_ provides several built-in codecs.


One could also use a third party encoding library such as libiconv to transcode to UTF-8.

Return C++ strings without conversion
-------------------------------------

If the data in a C++ ``std::string`` does not represent text and should be returned to Python as ``bytes``, then one can return the data as a ``py::bytes`` object.

.. code-block:: c++

    m.def("return_bytes",
        []() {
            std::string s("\xba\xd0\xba\xd0");  // Not valid UTF-8
            return py::bytes(s);  // Return the data without transcoding
        }
    );

.. code-block:: python

    >>> example.return_bytes()
    b'\xba\xd0\xba\xd0'


Wide character strings
======================

When a Python ``str`` is passed to a C++ function expecting ``std::u16string`` or ``std::u32string``, the ``str`` will be encoded to UTF-16 or UTF-32 respectively. 

.. warning::

    This may not work as described on Python 2.7 or Python 3.3 compiled with ``--enable-unicode=ucs2``.

References
==========

* `The Absolute Minimum Every Software Developer Absolutely, Positively Must Know About Unicode and Character Sets (No Excuses!) <https://www.joelonsoftware.com/2003/10/08/the-absolute-minimum-every-software-developer-absolutely-positively-must-know-about-unicode-and-character-sets-no-excuses/>`_
