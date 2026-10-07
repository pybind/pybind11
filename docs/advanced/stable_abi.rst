Python stable ABI (abi3)
########################

CPython's `stable ABI <https://docs.python.org/3/c-api/stable.html>`_ lets
one extension module run on every CPython release from the version it was
built for. pybind11 supports it from CPython 3.12 on: define
``Py_LIMITED_API`` to ``0x030C0000`` (or a newer version) and ship one
``*.abi3.so`` / ``*.pyd`` per platform instead of one module per Python
version.

Enabling it
===========

With CMake, add ``STABLE_ABI`` to ``pybind11_add_module`` (CMake 3.26+ and
the FindPython mode are required), or set ``PYBIND11_STABLE_ABI`` to make it
the default for a build tree:

.. code-block:: cmake

    pybind11_add_module(example STABLE_ABI example.cpp)

``PYBIND11_STABLE_ABI_VERSION`` (default ``3.12``) selects the targeted
version. ``PRECOMPILE`` can be combined with ``STABLE_ABI``; the precompiled
library is then compiled against the limited API too, and one build tree
cannot mix stable-ABI and regular precompiled modules.

With setuptools, pass ``py_limited_api=True`` (or a version such as
``"3.13"``) to ``Pybind11Extension``; the module is named ``*.abi3.so`` and
``Py_LIMITED_API`` is defined for you.

With any other build system, define ``Py_LIMITED_API=0x030C0000`` for every
translation unit, link no version-specific Python library (``python3.lib`` on
Windows), and name the module ``<name>.abi3.so`` (``<name>.pyd`` on Windows).

pybind11 rejects the combination with PyPy, GraalPy, and free-threaded
CPython at compile time: they have no stable ABI.

What changes under the stable ABI
=================================

The C++ API of pybind11 is the same; the differences are in what is
available and in how a few things are implemented.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Feature
     - Stable ABI
     - Notes
   * - Classes, functions, casters, STL, numpy, Eigen
     - supported
     -
   * - ``py::dynamic_attr()``
     - supported
     - The ``__dict__`` slot is appended to the instance (no
       ``Py_TPFLAGS_MANAGED_DICT``).
   * - Buffer protocol
     - supported
     - Through the ``Py_bf_*`` type slots.
   * - ``py::gil_scoped_acquire`` / ``release``
     - supported
     - ``PYBIND11_SIMPLE_GIL_MANAGEMENT`` is forced; ``disarm()`` and
       thread-state disassociation are not available.
   * - ``py::metaclass(handle)``
     - restricted
     - The metaclass must not define ``__new__``, and a metaclass less
       derived than ``pybind11_type`` resolves to ``pybind11_type``.
   * - ``py::custom_type_setup``
     - not available
     - Its callback receives a ``PyHeapTypeObject *``, which is opaque.
   * - ``pybind11/embed.h``
     - not available
     - Embedding uses non-limited initialization functions.
   * - ``pybind11/subinterpreter.h``
     - not available
     - Reads thread- and interpreter-state fields.
   * - ``pybind11/chrono.h``
     - not available
     - The ``datetime`` C API accesses struct fields.
   * - GIL-held assertions (``PYBIND11_ASSERT_GIL_HELD_INCREF_DECREF``)
     - not available
     - ``PyGILState_Check()`` is not part of the stable ABI.

ABI isolation
=============

A stable-ABI module and a regular module can be loaded in one process, but
they do not share pybind11's internal state: the internals ID carries a
``_stable`` tag, so each kind of module registers its own types and
instances. Types bound in one kind of module are opaque to the other, except
through the ``pybind11_conduit_v1`` protocol
(``include/pybind11/conduit/README.txt``), which bridges the two: the C++ ABI
is unchanged, so ``PYBIND11_PLATFORM_ABI_ID`` is the same.

Implementation notes and performance
====================================

Without access to CPython's struct layouts, pybind11 uses public functions
where it used to read fields directly. The costs are small and are confined
to specific operations:

* Type objects are created with ``PyType_FromMetaclass()`` and
  ``PyType_Spec`` (the ``PYBIND11_TYPE_CREATION_VIA_SPEC`` path, also
  available as an opt-in for regular builds on CPython 3.12+).
* Instance allocation and deallocation look up ``tp_alloc``/``tp_free`` with
  ``PyType_GetSlot()``; the ``__dict__`` of ``py::dynamic_attr()`` instances
  is found through the type's cached dictionary offset.
* Attribute access on bound *classes* (not instances) walks ``__mro__`` and
  the class dictionaries instead of using ``_PyType_Lookup()``.
* Methods are bound through a pybind11-provided ``instancemethod`` type and
  ``types.MethodType`` instead of ``PyInstanceMethod_Type`` /
  ``PyMethod_New()``.
* Tuple and list element access uses the checked function forms; type names
  in error messages are reconstructed from ``__module__`` and ``__name__``.
* ``py::eval``/``py::exec`` compile with ``Py_CompileString()`` and run with
  ``PyEval_EvalCode()``; ``py::eval_file`` reads the file through Python.

Modules built for the stable ABI check the interpreter version at import:
any CPython from the targeted version on is accepted.
