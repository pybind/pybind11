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


Known limitations
^^^^^^^^^^^^^^^^^

These are issues that are probably solvable, but have not been fixed yet. A
clean, well written patch would likely be accepted to solve them.

- ``py::cast<T>()`` does not keep type casters alive. A ``T`` that refers to
  data that its type caster owns, such as a container of ``char *``, is valid
  only as an argument of a bound function.
  `#2527 <https://github.com/pybind/pybind11/issues/2527>`_

Python 3.9.0 warning
^^^^^^^^^^^^^^^^^^^^

pybind11 older than 2.6.0 with Python 3.9.0 causes undefined behavior, which
usually shows as a crash at interpreter shutdown. The bug was `fixed in Python
3.9.1 <https://github.com/python/cpython/pull/22670>`_. pybind11 2.6.0 and newer
detect 3.9.0 at runtime and leak about 50 bytes each time a callback function is
garbage collected to avoid the bug. Use Python 3.9.1 or newer.
