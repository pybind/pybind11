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

- Classic pybind11 has the concept of "smart-pointer is holder".
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
Progressive mode will be easiest to use. For larger projects or projects
that integrate with third-party pybind11-based projects, the Conservative
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

There are three small differences compared to Classic pybind11:

- ``#include <pybind11/smart_holder.h>`` is used instead of
  ``#include <pybind11/pybind11.h>``.

- ``py::classh`` is used instead of ``py::class_``.

- The ``PYBIND11_SMART_HOLDER_TYPE_CASTERS(Foo)`` macro is needed.

To the 2nd bullet point, ``py::classh<Foo>`` is simply a shortcut for
``py::class_<Foo, py::smart_holder>``. The shortcut makes it possible to
switch to using ``py::smart_holder`` without disturbing the indentation of
existing code.

When migrating code that uses ``py::class_<Foo, std::shared_ptr<Foo>>``,
``std::shared_ptr<Foo>`` can be replaced with ``PYBIND11_SH_AVL(Foo)``,
which substitutes ``py::smart_holder`` in Conservative mode, but also allows
fallback to Classic mode by substituting ``std::shared_ptr<Foo>`` instead.

To the 3rd bullet point, the macro also needs to appear in other translation
units with pybind11 bindings that involve Python⇄C++ conversions for
`Foo`. This is the biggest inconvenience of the Conservative mode. Practically,
at a larger scale it is best to work with a pair of `.h` and `.cpp` files
for the bindings code, with the macros in the `.h` files.


Progressive mode
----------------

To work in Progressive mode:

- Add ``-DPYBIND11_USE_SMART_HOLDER_AS_DEFAULT`` to the compilation commands.

- Remove any ``std::shared_ptr<...>`` holders from existing ``py::class_``
  instantiations.

- Only if custom smart-pointers are used: the
  `PYBIND11_TYPE_CASTER_BASE_HOLDER` macro is needed [`example
  <https://github.com/pybind/pybind11/blob/2f624af1ac8571d603df2d70cb54fc7e2e3a356a/tests/test_multiple_inheritance.cpp#L72>`_].

Overall this is probably easier to work with than the Conservative mode, but

- the macro inconvenience is shifted from ``py::smart_holder`` to custom
  smart-pointers (but probably much more rarely needed).

- it will not interoperate with other extensions built against master or
  stable, or extensions built in Conservative mode (see the cross-module
  compatibility section below).


Transition from Conservative to Progressive mode
------------------------------------------------

This still has to be tried out more in practice, but in small-scale situations
it may be feasible to switch directly to Progressive mode in a break-fix
fashion. In large-scale situations it seems more likely that an incremental
approach is needed, which could mean incrementally converting ``py::class_``
to ``py::classh`` including addition of the macros, then flip the switch,
and convert ``py::classh`` back to ``py:class_`` combined with removal of the
macros if desired (at that point it will work equivalently either way). It
may be smart to delay the final cleanup step until all third-party projects
of interest have made the switch, because then the code will continue to
work in either mode.


Using py::smart_holder but with fallback to Classic pybind11
------------------------------------------------------------

For situations in which compatibility with Classic pybind11 (without
smart_holder) is needed for some period of time, fallback to Classic
mode can be enabled by copying the ``BOILERPLATE`` code block from
tests/test_classh_mock.cpp.

Fallback from Conservative to Classic mode could be viewed as
super-conservative mode.  The main idea is to enable use of ``py::classh``
and the associated ``PYBIND11_SMART_HOLDER_TYPE_CASTERS`` macro while still
being able to build the same code with Classic pybind11.

Fallback from Progressive to Classic mode is supported by the
``PYBIND11_SH_DEF(...)`` macro in the BOILERPLATE code block. "SH_DEF" is
short for "Smart_Holder if DEFault". The length of the macro is identical
by design to ``std::shared_ptr<...>``, to not disturb the indentation of
existing code.


Classic / Conservative / Progressive cross-module compatibility
---------------------------------------------------------------

Currently there are essentially three modes for building a pybind11 extension
module:

- Classic: pybind11 stable (e.g. v2.6.2) or current master branch.

- Conservative: pybind11 smart_holder branch.

- Progressive: pybind11 smart_holder branch with
  ``-DPYBIND11_USE_SMART_HOLDER_AS_DEFAULT``.

In environments that mix extension modules built with different modes,
this is the compatibility matrix for ``py::class_``-wrapped types:

.. list-table:: Compatibility matrix
   :widths: auto
   :header-rows: 2

   * -
     -
     -
     - Module 2
     -
   * -
     -
     - Classic
     - Conservative
     - Progressive
   * -
     - **Classic**
     - full
     - one-and-a-half-way
     - isolated
   * - **Module 1**
     - **Conservative**
     - one-and-a-half-way
     - full
     - isolated
   * -
     - **Progressive**
     - isolated
     - isolated
     - full

Mixing Classic+Progressive or Conservative+Progressive is very easy to
understand: the extension modules are essentially completely isolated from
each other. This is in fact just the same as using pybind11 versions with
differing `"internals version"
<https://github.com/pybind/pybind11/blob/114be7f4ade0ad798cd4c7f5d65ebe4ba8bd892d/include/pybind11/detail/internals.h#L95>`_
in the past. While this is easy to understand, there is also no incremental
transition path between Classic and Progressive.

The Conservative mode enables incremental transitions, but at the cost of
more complexity. Types wrapped in a Classic module are fully compatible with
a Conservative module. However, a type wrapped in a Conservative module is
compatible with a Classic module only if ``py::smart_holder`` is **not** used
(for that type). A type wrapped with ``py::smart_holder`` is incompatible with
a Classic module. This is an important pitfall to keep in mind: attempts to use
``py::smart_holder``-wrapped types in a Classic module will lead to undefined
runtime behavior, such as a SEGFAULT. This is a more general flavor of the
long-standing issue `#1138 <https://github.com/pybind/pybind11/issues/1138>`_,
often referred to as "holder mismatch". It is important to note that the
pybind11 smart_holder branch solves the smart-pointer interoperability issue,
but not the more general holder mismatch issue. — Unfortunately the existing
pybind11 internals do not track holder runtime type information, therefore
the holder mismatch issue cannot be solved in a fashion that would allow
an incremental transition, which is the whole point of the Conservative
mode. Please proceed with caution.

Another pitfall worth pointing out specifically, although it follows
from the previous: mixing base and derived classes between Classic and
Conservative modules means that neither the base nor the derived class can
use ``py::smart_holder``.


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

This is the only difference compared to Classic pybind11. A fairly
minimal but complete example is tests/test_class_sh_trampoline_unique_ptr.cpp.


Ideas for the long-term
-----------------------

The macros are clearly an inconvenience in many situations. Highly
speculative: to avoid the need for the macros, a potential approach would
be to combine the Classic implementation (``type_caster_base``) with
the ``smart_holder_type_caster``, but this will probably be very messy and
not great as a long-term solution. The ``type_caster_base`` code is very
complex already. A more maintainable approach long-term could be to work
out and document a smart_holder-based solution for custom smart-pointers
in pybind11 version ``N``, then purge ``type_caster_base`` in version
``N+1``. #HelpAppreciated.


GitHub testing of PRs against the smart_holder branch
-----------------------------------------------------

PRs against the smart_holder branch need to be tested in both
modes (Conservative, Progressive), with the only difference that
``PYBIND11_USE_SMART_HOLDER_AS_DEFAULT`` is defined for Progressive mode
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
   git push --force-with-lease  # This will trigger the GitHub Actions for the Progressive mode.

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
