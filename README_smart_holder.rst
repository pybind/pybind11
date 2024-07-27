==============================
pybind11 — smart_holder branch
==============================


Overview
========

- The smart_holder branch is a strict superset of the pybind11 master branch.
  Everything that works with the master branch is expected to work exactly the
  same with the smart_holder branch.

- Activating the smart_holder functionality for a given C++ type ``T`` is as
  easy as changing ``py::class_<T>`` to ``py::classh<T>`` in client code.

- The ``py::classh<T>`` functionality includes

  * support for **two-way** Python/C++ conversions for both
    ``std::unique_ptr<T>`` and ``std::shared_ptr<T>`` **simultaneously**.
    — In contrast, ``py::class_<T>`` only supports one-way C++-to-Python
    conversions for ``std::unique_ptr<T>``, or alternatively two-way
    Python/C++ conversions for ``std::shared_ptr<T>``, which then excludes
    the one-way C++-to-Python ``std::unique_ptr<T>`` conversions (this
    manifests itself through undefined runtime behavior).

  * passing a Python object back to C++ via ``std::unique_ptr<T>``, safely
    **disowning** the Python object.

  * safely passing `"trampoline"
    <https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python>`_
    objects (objects with C++ virtual function overrides implemented in
    Python) via ``std::unique_ptr<T>`` or ``std::shared_ptr<T>`` back to C++:
    associated Python objects are automatically kept alive for the lifetime
    of the smart-pointer.

Note: As of `PR #5257 <https://github.com/pybind/pybind11/pull/5257>`_
the smart_holder functionality is fully baked into pybind11.
Prior to PR #5257 the smart_holder implementation was an "add-on", which made
it necessary to use a ``PYBIND11_SMART_HOLDER_TYPE_CASTERS`` macro. This macro
still exists for backward compatibility, but is now a no-op. The trade-off
for this convenience is that the ``PYBIND11_INTERNALS_VERSION`` needed to be
changed. Consequently, Python extension modules built with the smart_holder
branch no longer interoperate with extension modules built with the pybind11
master branch. If cross-extension-module interoperability is required, all
extension modules involved must be built with the smart_holder branch.
— Probably, most extension modules do not require cross-extension-module
interoperability, but exceptions to this are quite common.


What is fundamentally different?
--------------------------------

- Classic pybind11 has the concept of "smart-pointer is holder".
  Interoperability between smart-pointers is completely missing. For example,
  with ``py::class_<T, std::shared_ptr<T>>``, ``return``-ing a
  ``std::unique_ptr<T>`` leads to undefined runtime behavior
  (`#1138 <https://github.com/pybind/pybind11/issues/1138>`_).
  A `systematic analysis can be found here
  <https://github.com/pybind/pybind11/pull/2672#issuecomment-748392993>`_.

- ``py::smart_holder`` has a richer concept in comparison, with well-defined
  runtime behavior in all situations. ``py::smart_holder`` "knows" about both
  ``std::unique_ptr<T>`` and ``std::shared_ptr<T>``, and how they interoperate.


What motivated the development of the smart_holder code?
--------------------------------------------------------

- The original context was retooling of `PyCLIF
  <https://github.com/google/clif/>`_, to use pybind11 underneath,
  instead of directly targeting the Python C API. Essentially the smart_holder
  branch is porting established PyCLIF functionality into pybind11. (However,
  this work also led to bug fixes in PyCLIF.)


Installation
============

Currently ``git clone`` is the only option. We do not have released packages.

.. code-block:: bash

   git clone --branch smart_holder https://github.com/pybind/pybind11.git

Everything else is exactly identical to using the default (master) branch.


Trampolines and std::unique_ptr
-------------------------------

A pybind11 `"trampoline"
<https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python>`_
is a C++ helper class with virtual function overrides that transparently
call back from C++ into Python. To enable safely passing a ``std::unique_ptr``
to a trampoline object between Python and C++, the trampoline class must
inherit from ``py::trampoline_self_life_support``, for example:

.. code-block:: cpp

   class PyAnimal : public Animal, public py::trampoline_self_life_support {
       ...
   };

This is the only difference compared to classic pybind11. A fairly
minimal but complete example is tests/test_class_sh_trampoline_unique_ptr.cpp.


Related links
=============

* The smart_holder branch addresses issue
  `#1138 <https://github.com/pybind/pybind11/issues/1138>`_ and
  the ten issues enumerated in the `description of PR 2839
  <https://github.com/pybind/pybind11/pull/2839#issue-564808678>`_.

* `Description of PR #2672
  <https://github.com/pybind/pybind11/pull/2672#issue-522688184>`_, from which
  the smart_holder branch was created.

* Small `slide deck
  <https://docs.google.com/presentation/d/1r7auDN0x-b6uf-XCvUnZz6z09raasRcCHBMVDh7PsnQ/>`_
  presented in meeting with pybind11 maintainers on Feb 22, 2021. Slides 5
  and 6 show performance comparisons. (These are outdated but probably not far off.)
