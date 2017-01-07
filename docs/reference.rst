.. _reference:

.. warning::

    Please be advised that the reference documentation discussing pybind11
    internals is currently incomplete. Please refer to the previous sections
    and the pybind11 header files for the nitty gritty details.

Reference
#########

Macros
======

.. doxygendefine:: PYBIND11_PLUGIN

.. _core_types:

Convenience classes for arbitrary Python types
==============================================

Common member functions
-----------------------

.. doxygenclass:: object_api
    :members:

Without reference counting
--------------------------

.. doxygenclass:: handle
    :members:

With reference counting
-----------------------

.. doxygenclass:: object
    :members:

.. doxygenfunction:: reinterpret_borrow

.. doxygenfunction:: reinterpret_steal

Convenience classes for specific Python types
=============================================

.. doxygenclass:: module
    :members:

.. _extras:

Passing extra arguments to the def function
===========================================

.. doxygenstruct:: arg
    :members:

.. doxygenstruct:: arg_v
    :members:
