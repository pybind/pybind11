# -*- coding: utf-8 -*-
# pybind11 equivalent of Boost.Python test:
# https://github.com/rwgk/rwgk_tbx/blob/6c9a6d6bc72d5c1b8609724433259c5b47178680/tst_cpp_base_py_derived.py
# See also: https://github.com/pybind/pybind11/issues/1333 (this was the starting point)
import pytest

from pybind11_tests import cpp_base_py_derived as m


class drvd(m.base):

  def __init__(self, _num = 200):
    super().__init__()
    self._drvd_num = _num

  def get_num(self):
    return self._drvd_num

  def clone(self):
    return drvd(250)


def test_base():
  b = m.base()
  assert b.get_num() == 100
  assert m.get_num(b) == 100
  bc = b.clone()
  assert bc.get_num() == 150
  assert m.clone_get_num(b) == 103157


def test_drvd():
  d = drvd()
  assert d.get_num() == 200
  assert m.get_num(d) == 200
  dc = d.clone()
  assert dc.get_num() == 250
  assert m.clone_get_num(d) == 203257
