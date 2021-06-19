# -*- coding: utf-8 -*-

from pybind11_tests import class_sh_shared_ptr_copy_move as m


def test_avl_copy():
    m.test_avl_copy()


def test_def_copy():
    m.test_def_copy()


def test_avl_move():
    m.test_avl_move()


def test_def_move():
    m.test_def_move()
