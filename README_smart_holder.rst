==============================
pybind11 — smart_holder branch
==============================


Overview
========

- The smart_holder git branch is a strict superset of the master
  branch. Everything that works on master is expected to work exactly the same
  with the smart_holder branch.

- **Smart-pointer interoperability** (``std::unique_ptr``, ``std::shared_ptr``)
  is implemented as an **add-on**.

- The add-on also supports
    * passing a Python object back to C++ via ``std::unique_ptr``, safely
      **disowning** the Python object.
    * safely passing `"trampoline"
      <https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python>`_
      objects (objects with C++ virtual function overrides implemented in
      Python) via ``std::unique_ptr`` or ``std::shared_ptr`` back to C++:
      associated Python objects are automatically kept alive for the lifetime
      of the smart-pointer.

- The smart_holder branch can be used in two modes:
    * **Conservative mode**: ``py::class_`` works exactly as on master.
      ``py::classh`` uses ``py::smart_holder``.
    * **Progressive mode**: ``py::class_`` uses ``py::smart_holder``
      (i.e. ``py::smart_holder`` is the default holder).


What is fundamentally different?
--------------------------------

- Traditional pybind11 has the concept of "smart-pointer is holder".
  Interoperability between smart-pointers is completely missing. For
  example, when using ``std::shared_ptr`` as holder, ``return``-ing
  a ``std::unique_ptr`` leads to undefined runtime behavior
  (`#1138 <https://github.com/pybind/pybind11/issues/1138>`_). A
  `systematic analysis is here <https://github.com/pybind/pybind11/pull/2672#issuecomment-748392993>`_.

- ``py::smart_holder`` has a richer concept in comparison, with well-defined
  runtime behavior. The holder "knows" about both ``std::unique_ptr`` and
  ``std::shared_ptr`` and how they interoperate.

- Caveat (#HelpAppreciated): currently the ``smart_holder`` branch does
  not have a well-lit path for including interoperability with custom
  smart-pointers. It is expected to be a fairly obvious extension of the
  ``smart_holder`` implementation, but will depend on the exact specifications
  of each custom smart-pointer type (generalizations are very likely possible).


What motivated the development of the smart_holder code?
--------------------------------------------------------

- Necessity is the mother. The bigger context is the ongoing retooling of
  `PyCLIF <https://github.com/google/clif/>`_, to use pybind11 underneath
  instead of directly targeting the Python C API. Essentially, the smart_holder
  branch is porting established PyCLIF functionality into pybind11.


Installation
============

Currently ``git clone`` is the only option. We do not have released packages.

.. code-block:: bash

   git clone --branch smart_holder https://github.com/pybind/pybind11.git

Everything else is exactly identical to using the default (master) branch.


Conservative or Progressive mode?
=================================

It depends. To a first approximation, for a stand-alone, new project, the
progressive mode will be easiest to use. For larger projects or projects
that integrate with third-party pybind11-based projects, the conservative
mode may be more practical, at least initially, although it comes with the
disadvantage of having to use the ``PYBIND11_SMART_HOLDER_TYPE_CASTERS`` macro.


Conservative mode
-----------------

Here is a minimal example for wrapping a C++ type with ``py::smart_holder`` as
holder:

.. code-block:: cpp

   #include <pybind11/smart_holder.h>

   struct Foo {};

   PYBIND11_SMART_HOLDER_TYPE_CASTERS(Foo)

   PYBIND11_MODULE(example_bindings, m) {
       namespace py = pybind11;
       py::classh<Foo>(m, "Foo");
   }

There are three small differences compared to traditional pybind11:

- ``#include <pybind11/smart_holder.h>`` is used instead of
  ``#include <pybind11/pybind11.h>``.

- ``py::classh`` is used instead of ``py::class_``.

- The ``PYBIND11_SMART_HOLDER_TYPE_CASTERS(Foo)`` macro is needed.

To the 2nd bullet point, ``py::classh<Foo>`` is simply a shortcut for
``py::class_<Foo, py::smart_holder>``. The shortcut makes it possible to
switch to using ``py::smart_holder`` without messing up the indentation of
existing code. However, when migrating code that uses ``py::class_<Foo,
std::shared_ptr<Foo>>``, currently ``std::shared_ptr<Foo>`` needs to be
removed manually when switching to ``py::classh`` (#HelpAppreciated this
could probably be avoided with a little bit of template metaprogramming).

To the 3rd bullet point, the macro also needs to appear in other translation
units with pybind11 bindings that involve Python⇄C++ conversions for
`Foo`. This is the biggest inconvenience of the conservative mode. Practially,
at a larger scale it is best to work with a pair of `.h` and `.cpp` files
for the bindings code, with the macros in the `.h` files.


Progressive mode
----------------

To work in progressive mode:

- Add ``-DPYBIND11_USE_SMART_HOLDER_AS_DEFAULT`` to the compilation commands.

- Remove any ``std::shared_ptr<...>`` holders from existing ``py::class_``
  instantiations (#HelpAppreciated this could probably be avoided with a little
  bit of template metaprogramming).

- Only if custom smart-pointers are used: the
  `PYBIND11_TYPE_CASTER_BASE_HOLDER` macro is needed [`example
  <https://github.com/pybind/pybind11/blob/2f624af1ac8571d603df2d70cb54fc7e2e3a356a/tests/test_multiple_inheritance.cpp#L72>`_].

Overall this is probably easier to work with than the conservative mode, but

- the macro inconvenience is shifted from ``py::smart_holder`` to custom
  smart-pointers (but probably much more rarely needed).

- it will not interoperate with other extensions built against master or
  stable, or extensions built in conservative mode.


Transition from conservative to progressive mode
------------------------------------------------

This still has to be tried out more in practice, but in small-scale situations
it may be feasible to switch directly to progressive mode in a break-fix
fashion. In large-scale situations it seems more likely that an incremental
approach is needed, which could mean incrementally converting ``py::class_``
to ``py::classh`` including addition of the macros, then flip the switch,
and convert ``py::classh`` back to ``py:class_`` combined with removal of the
macros if desired (at that point it will work equivalently either way). It
may be smart to delay the final cleanup step until all third-party projects
of interest have made the switch, because then the code will continue to
work in either mode.


Ideas for the long-term
-----------------------

The macros are clearly an inconvenience in many situations. Highly
speculative: to avoid the need for the macros, a potential approach would
be to combine the traditional implementation (``type_caster_base``) with
the ``smart_holder_type_caster``, but this will probably be very messy and
not great as a long-term solution. The ``type_caster_base`` code is very
complex already. A more maintainable approach long-term could be to work
out and document a smart_holder-based solution for custom smart-pointers
in pybind11 version ``N``, then purge ``type_caster_base`` in version
``N+1``. #HelpAppreciated.


GitHub testing of PRs against the smart_holder branch
-----------------------------------------------------

PRs against the smart_holder branch need to be tested in both
modes (conservative, progressive), with the only difference that
``PYBIND11_USE_SMART_HOLDER_AS_DEFAULT`` is defined for progressive mode
testing. Currently this is handled simply by creating a secondary PR with a
one-line change in ``include/pybind11/detail/smart_holder_sfinae_hooks_only.h``
(as in e.g. `PR #2879 <https://github.com/pybind/pybind11/pull/2879>`_). It
will be best to mark the secondary PR as Draft. Often it is convenient to reuse
the same secondary PR for a series of primary PRs, simply by rebasing on a
primary PR as needed:

.. code-block:: bash

   git checkout -b sh_primary_pr
   # Code development ...
   git push  # Create a PR as usual, selecting smart_holder from the branch pulldown.
   git checkout sh_secondary_pr
   git rebase -X theirs sh_primary_pr
   git diff  # To verify that the one-line change in smart_holder_sfinae_hooks_only.h is the only diff.
   git push --force-with-lease  # This will trigger the GitHub Actions for the progressive mode.

The second time through this will only take a minute or two.


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
  and 6 show performance comparisons.
