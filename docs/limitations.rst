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

- Intel 20.2 is currently having an issue with the test suite.
  `#2573 <https://github.com/pybind/pybind11/pull/2573>`_

- Debug mode Python does not support 1-5 tests in the test suite currently.
  `#2422 <https://github.com/pybind/pybind11/pull/2422>`_

Known limitations
^^^^^^^^^^^^^^^^^

These are issues that are probably solvable, but have not been fixed yet. A
clean, well written patch would likely be accepted to solve them.

- Type casters are not kept alive recursively.
  `#2527 <https://github.com/pybind/pybind11/issues/2527>`_
  One consequence is that containers of ``char *`` are currently not supported.
  `#2245 <https://github.com/pybind/pybind11/issues/2245>`_

- The ``cpptest`` does not run on Windows with Python 3.8 or newer, due to DLL
  loader changes. User code that is correctly installed should not be affected.
  `#2560 <https://github.com/pybind/pybind11/issue/2560>`_
