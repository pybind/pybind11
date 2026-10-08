Python stable ABI (abi3, abi3t)
###############################

CPython's `stable ABI <https://docs.python.org/3/c-api/stable.html>`_ lets
one extension module run on every CPython release from the version it was
built for. pybind11 supports it from CPython 3.12 on: define
``Py_LIMITED_API`` to ``0x030C0000`` (or a newer version) and ship one
``*.abi3.so`` / ``*.pyd`` per platform instead of one module per Python
version.

CPython 3.15 adds a second stable ABI, ``abi3t`` (:pep:`803`). An ``abi3t``
module loads on every free-threaded *and* GIL-enabled CPython from 3.15 on, so
one ``*.abi3t.so`` per platform covers both builds. pybind11 supports it too:
define ``Py_TARGET_ABI3T`` to ``0x030F0000`` (or a newer version). Headers of
either build work. With free-threaded headers, ``Py_LIMITED_API`` alone also
selects ``abi3t``, because free-threaded CPython has no ``abi3``. See `abi3t`_
below for what differs.

Enabling it
===========

With CMake, add ``STABLE_ABI`` to ``pybind11_add_module`` (CMake 3.26+ and
the FindPython mode are required), or set ``PYBIND11_STABLE_ABI`` to make it
the default for a build tree:

.. code-block:: cmake

    pybind11_add_module(example STABLE_ABI example.cpp)

``PYBIND11_STABLE_ABI_VERSION`` (default ``3.12``, raised to ``3.15`` for
``abi3t``) selects the targeted version. On free-threaded Python the modules
are always ``abi3t``. Set ``PYBIND11_ABI3T`` to ``ON`` to also build ``abi3t``
modules with GIL-enabled Python:

.. code-block:: bash

    cmake -S . -B build -DPYBIND11_STABLE_ABI=ON -DPYBIND11_ABI3T=ON

``PRECOMPILE`` can be combined with ``STABLE_ABI``; the precompiled library
is then compiled against the limited API too, and one build tree cannot mix
stable-ABI and regular precompiled modules.

With setuptools, pass ``py_limited_api=True`` (or a version such as
``"3.13"``) to ``Pybind11Extension``; the module is named ``*.abi3.so``
and ``Py_LIMITED_API`` is defined for you. Add a ``t`` (``"3.15t"``) for
``abi3t``, which is always used on free-threaded Python; ``Py_TARGET_ABI3T``
is then defined and the module is named ``*.abi3t.so``. With GIL-enabled
Python, use ``build_ext`` from ``pybind11.setup_helpers`` to get that name.

With any other build system, define ``Py_LIMITED_API=0x030C0000`` for every
translation unit, link no version-specific Python library (``python3.lib`` on
Windows), and name the module ``<name>.abi3.so`` (``<name>.pyd`` on Windows).
For ``abi3t``, define ``Py_TARGET_ABI3T=0x030F0000`` instead, link
``python3t.lib`` on Windows, and name the module ``<name>.abi3t.so``.

pybind11 rejects the combination with PyPy, GraalPy, and free-threaded
CPython before 3.15 at compile time: they have no stable ABI.

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
     - Under abi3t, ``numpy.h`` needs NumPy 2.5+ at run time (it reads
       array fields through NumPy's abi3t accessors).
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
     - supported
     - The ``datetime`` C API is not in the stable ABI; fields are read as
       attributes and objects are built by calling the ``datetime`` types.
   * - GIL-held assertions (``PYBIND11_ASSERT_GIL_HELD_INCREF_DECREF``)
     - not available
     - ``PyGILState_Check()`` is not part of the stable ABI.

abi3t
=====

Under abi3t, ``PyObject`` and ``PyModuleDef`` are incomplete types and
``PyMutex`` and ``PyUnstable_TryIncRef()`` are not available. pybind11 then:

* stores its per-instance data as :pep:`697` type data
  (``PyObject_GetTypeData()``) instead of embedding ``PyObject_HEAD``;
* exports the module through the :pep:`793` ``PyModExport_<name>`` hook
  instead of ``PyInit_<name>``, so ``PYBIND11_MODULE`` is unchanged but
  ``py::module_::create_extension_module()`` is not available;
* locks its internals with critical sections on a private object, and keeps
  one weak reference per registered instance so that the instance registry
  can hand out strong references safely (``PyWeakref_GetRef()``).

``pybind11/numpy.h`` reads NumPy's object fields through the accessors NumPy
2.5 added for abi3t, so older NumPy versions are rejected at run time. The
internals tag is ``_stable_ft``: an abi3t module and an abi3 module loaded
into the same GIL-enabled interpreter do not share internals (see below).

ABI isolation
=============

A stable-ABI module and a regular module can be loaded in one process, but
they do not share pybind11's internal state: the internals ID carries a
``_stable`` (abi3) or ``_stable_ft`` (abi3t) tag, so each kind of module
registers its own types and
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
  ``PyType_Spec``. This is also the default for regular builds on CPython
  3.12+; define ``PYBIND11_TYPE_CREATION_VIA_SPEC=0`` (CMake:
  ``-DPYBIND11_TYPE_CREATION_VIA_SPEC=OFF``) to use the previous path, which
  fills in the ``PyHeapTypeObject`` fields directly.
* Instance allocation and deallocation look up ``tp_alloc``/``tp_free`` with
  ``PyType_GetSlot()``; the ``__dict__`` of ``py::dynamic_attr()`` instances
  is found through the type's cached dictionary offset.
* Attribute *assignment* on bound classes (``Type.static_prop = value``) walks
  ``__mro__`` and the class dictionaries instead of using ``_PyType_Lookup()``;
  attribute reads use the metaclass default and are not affected.
* Methods are bound through a pybind11-provided ``instancemethod`` type and
  ``types.MethodType`` instead of ``PyInstanceMethod_Type`` /
  ``PyMethod_New()``.
* Tuple and list element access uses the checked function forms; type names
  in error messages are reconstructed from ``__module__`` and ``__name__``.
* ``py::eval``/``py::exec`` compile with ``Py_CompileString()`` and run with
  ``PyEval_EvalCode()``; ``py::eval_file`` reads the file through Python.

Modules built for the stable ABI check the interpreter version at import:
any CPython from the targeted version on is accepted.
