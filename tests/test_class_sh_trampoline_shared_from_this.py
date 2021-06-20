# -*- coding: utf-8 -*-

import env  # noqa: F401

import pybind11_tests.class_sh_trampoline_shared_from_this as m

import gc
import weakref


class PySft(m.Sft):
    pass


def test_pass_shared_ptr():
    obj = PySft("PySft")
    assert obj.history == "PySft"
    assert obj.use_count() in [2, -1]  # TODO: Be smarter/stricter.
    m.pass_shared_ptr(obj)
    assert obj.history == "PySft_PassSharedPtr"
    assert obj.use_count() in [2, -1]
    m.pass_shared_ptr(obj)
    assert obj.history == "PySft_PassSharedPtr_PassSharedPtr"
    assert obj.use_count() in [2, -1]


def test_pass_shared_ptr_while_stashed():
    obj = PySft("PySft")
    obj_wr = weakref.ref(obj)
    stash1 = m.SftSharedPtrStash(1)
    stash1.Add(obj)
    assert obj.history == "PySft_Stash1Add"
    assert obj.use_count() in [2, -1]
    m.pass_shared_ptr(obj)
    assert obj.history == "PySft_Stash1Add_PassSharedPtr"
    assert obj.use_count() in [2, -1]
    stash2 = m.SftSharedPtrStash(2)
    stash2.Add(obj)
    assert obj.history == "PySft_Stash1Add_PassSharedPtr_Stash2Add"
    assert obj.use_count() in [2, -1]
    assert stash2.history(0) == "PySft_Stash1Add_PassSharedPtr_Stash2Add"
    assert stash2.use_count(0) == 1  # TODO: this is not great.
    stash2.Add(obj)
    assert obj.history == "PySft_Stash1Add_PassSharedPtr_Stash2Add_Stash2Add"
    assert obj.use_count() in [2, -1]
    assert stash1.use_count(0) == 1
    assert stash1.history(0) == "PySft_Stash1Add_PassSharedPtr_Stash2Add_Stash2Add"
    assert stash2.use_count(0) == 1
    assert stash2.history(0) == "PySft_Stash1Add_PassSharedPtr_Stash2Add_Stash2Add"
    assert stash2.use_count(1) == 1
    assert stash2.history(1) == "PySft_Stash1Add_PassSharedPtr_Stash2Add_Stash2Add"
    del obj
    assert stash2.use_count(0) == 1
    assert stash2.history(0) == "PySft_Stash1Add_PassSharedPtr_Stash2Add_Stash2Add"
    assert stash2.use_count(1) == 1
    assert stash2.history(1) == "PySft_Stash1Add_PassSharedPtr_Stash2Add_Stash2Add"
    del stash2
    gc.collect()
    assert obj_wr() is not None
    assert stash1.history(0) == "PySft_Stash1Add_PassSharedPtr_Stash2Add_Stash2Add"
    del stash1
    gc.collect()
    if not env.PYPY:
        assert obj_wr() is None


def test_pass_shared_ptr_while_stashed_with_shared_from_this():
    obj = PySft("PySft")
    obj_wr = weakref.ref(obj)
    stash1 = m.SftSharedPtrStash(1)
    stash1.AddSharedFromThis(obj)
    assert obj.history == "PySft_Stash1AddSharedFromThis"
    assert stash1.use_count(0) == 2
    stash1.AddSharedFromThis(obj)
    assert obj.history == "PySft_Stash1AddSharedFromThis_Stash1AddSharedFromThis"
    assert stash1.use_count(0) == 3
    assert stash1.use_count(1) == 3
    del obj
    del stash1
    gc.collect()
    if not env.PYPY:
        assert obj_wr() is None
