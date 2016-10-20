Eigen
=====

`Eigen <http://eigen.tuxfamily.org>`_ is C++ header-based library for dense and
sparse linear algebra. Due to its popularity and widespread adoption, pybind11
provides transparent conversion support between Eigen and Scientific Python linear
algebra data types.

Specifically, when including the optional header file :file:`pybind11/eigen.h`,
pybind11 will automatically and transparently convert

1. Static and dynamic Eigen dense vectors and matrices to instances of
   ``numpy.ndarray`` (and vice versa).

2. Returned matrix expressions such as blocks (including columns or rows) and
   diagonals will be converted to ``numpy.ndarray`` of the expression
   values.

3. Returned matrix-like objects such as Eigen::DiagonalMatrix or
   Eigen::SelfAdjointView will be converted to ``numpy.ndarray`` containing the
   expressed value.

4. Eigen sparse vectors and matrices to instances of
   ``scipy.sparse.csr_matrix``/``scipy.sparse.csc_matrix`` (and vice versa).

This makes it possible to bind most kinds of functions that rely on these types.
One major caveat are functions that take Eigen matrices *by reference* and modify
them somehow, in which case the information won't be propagated to the caller.

.. code-block:: cpp

    /* The Python bindings of these functions won't replicate
       the intended effect of modifying the function arguments */
    void scale_by_2(Eigen::Vector3f &v) {
        v *= 2;
    }
    void scale_by_2(Eigen::Ref<Eigen::MatrixXd> &v) {
        v *= 2;
    }

To see why this is, refer to the section on :ref:`opaque` (although that
section specifically covers STL data types, the underlying issue is the same).
The :ref:`numpy` sections discuss an efficient alternative for exposing the
underlying native Eigen types as opaque objects in a way that still integrates
with NumPy and SciPy.

.. seealso::

    The file :file:`tests/test_eigen.cpp` contains a complete example that
    shows how to pass Eigen sparse and dense data types in more detail.
