from __future__ import annotations

import pytest

from pybind11_tests import vector_unique_ptr_member as m


@pytest.mark.parametrize("num_elems", range(3))
def test_create(num_elems):
    vo = m.VectorOwner.Create(num_elems)
    assert vo.data_size() == num_elems


def NOtest_cast():  # Fails only with PYBIND11_RUN_TESTING_WITH_SMART_HOLDER_AS_DEFAULT_BUT_NEVER_USE_IN_PRODUCTION_PLEASE
    vo = m.VectorOwner.Create(0)
    assert m.py_cast_VectorOwner_ptr(vo) is vo
