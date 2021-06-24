# -*- coding: utf-8 -*-

from pybind11_tests import class_sh_shared_ptr_copy_move as m


def test_shptr_copy():
    txt = m.test_ShPtr_copy()[0].get_history()
    assert txt == "FooShPtr_copy"


def test_smhld_copy():
    txt = m.test_SmHld_copy()[0].get_history()
    assert txt == "FooSmHld_copy"


def test_shptr_move():
    txt = m.test_ShPtr_move()[0].get_history()
    assert txt == "FooShPtr_move"


def test_smhld_move():
    txt = m.test_SmHld_move()[0].get_history()
    assert txt == "FooSmHld_move"
