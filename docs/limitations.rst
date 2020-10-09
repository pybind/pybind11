Limitations
###########

Design choices
^^^^^^^^^^^^^^

pybind11 strives to be a general solution to binding generation, but it also has
certain limitations:

- pybind11 casts away ``const``-ness in function arguments and return values.
  This is in line with the Python language, which has no concept of ``const``
  values. This means that some additional care is needed to avoid bugs that
  would be caught by the type checker in a traditional C++ program.

- The NumPy interface ``pybind11::array`` greatly simplifies accessing
  numerical data from C++ (and vice versa), but it's not a full-blown array
  class like ``Eigen::Array`` or ``boost.multi_array``. ``Eigen`` objects are
  directly supported, however, with ``pybind11/eigen.h``.

Large but useful features could be implemented in pybind11 but would lead to a
significant increase in complexity. Pybind11 strives to be simple and compact.
Users who require large new features are encouraged to write an extension to
pybind11; see `pybind11_json <https://github.com/pybind/pybind11_json>`_ for an
example.


Known bugs
^^^^^^^^^^

These are issues that hopefully will one day be fixed, but currently are
unsolved. If you know how to help with one of these issues, contributions
are welcome!

- The test suite currently segfaults on macOS and Python 3.9.0 when exiting the
  interpreter.  This was suspected to be related to the cross module GIL code,
  but could be a broader Python 3.9.0 issue.
  `#2558 <https://github.com/pybind/pybind11/issues/2558>`_

- The ``cpptest`` does not run on Windows with Python 3.8 or newer, due to DLL
  loader changes. User code that is correctly installed should not be affected.
  `#2560 <https://github.com/pybind/pybind11/pull/2560>`_

- There may be a rare issue with leakage under some compilers, exposed by
  adding an unrelated test to the test suite.
  `#2335 <https://github.com/pybind/pybind11/pull/2335>`_

Known limitations
^^^^^^^^^^^^^^^^^

These are issues that are probably solvable, but have not been fixed yet. A
clean, well written patch would likely be accepted to solve them.

- Type casters are not kept alive recursively.
  `#2527 <https://github.com/pybind/pybind11/issues/2527>`_
  One consequence is that containers of ``char *`` are currently not supported.
  `#2245 <https://github.com/pybind/pybind11/issues/2245>`_
