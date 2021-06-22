# -*- coding: utf-8 -*-
import pytest

import pybind11_tests.class_sh_trampoline_shared_from_this as m


class PySft(m.Sft):
    pass


def test_release_and_immediate_reclaim():
    obj = PySft("PySft")
    assert obj.history == "PySft"
    assert obj.use_count() == 1
    assert m.pass_shared_ptr(obj) == 2
    assert obj.history == "PySft_PassSharedPtr"
    assert obj.use_count() == 1
    assert m.pass_shared_ptr(obj) == 2
    assert obj.history == "PySft_PassSharedPtr_PassSharedPtr"
    assert obj.use_count() == 1

    obj = PySft("")
    while True:
        m.pass_shared_ptr(obj)
        assert obj.history == ""
        assert obj.use_count() == 1
        break  # Comment out for manual leak checking (use `top` command).


def test_release_to_cpp_stash():
    obj = PySft("PySft")
    stash1 = m.SftSharedPtrStash(1)
    stash1.Add(obj)
    assert obj.history == "PySft_Stash1Add"
    assert obj.use_count() == 1
    assert stash1.history(0) == "PySft_Stash1Add"
    assert stash1.use_count(0) == 1  # obj does NOT own the shared_ptr anymore.
    assert m.pass_shared_ptr(obj) == 3
    assert obj.history == "PySft_Stash1Add_PassSharedPtr"
    assert obj.use_count() == 1
    assert stash1.history(0) == "PySft_Stash1Add_PassSharedPtr"
    assert stash1.use_count(0) == 1
    stash2 = m.SftSharedPtrStash(2)
    stash2.Add(obj)
    assert obj.history == "PySft_Stash1Add_PassSharedPtr_Stash2Add"
    assert obj.use_count() == 2
    assert stash2.history(0) == "PySft_Stash1Add_PassSharedPtr_Stash2Add"
    assert stash2.use_count(0) == 2
    stash2.Add(obj)
    exp_oh = "PySft_Stash1Add_PassSharedPtr_Stash2Add_Stash2Add"
    assert obj.history == exp_oh
    assert obj.use_count() == 3
    assert stash1.history(0) == exp_oh
    assert stash1.use_count(0) == 3
    assert stash2.history(0) == exp_oh
    assert stash2.use_count(0) == 3
    assert stash2.history(1) == exp_oh
    assert stash2.use_count(1) == 3
    del obj
    assert stash2.history(0) == exp_oh
    assert stash2.use_count(0) == 3
    assert stash2.history(1) == exp_oh
    assert stash2.use_count(1) == 3
    del stash2
    assert stash1.history(0) == exp_oh
    assert stash1.use_count(0) == 1


def test_release_to_cpp_stash_leak():
    obj = PySft("")
    while True:
        stash1 = m.SftSharedPtrStash(1)
        stash1.Add(obj)
        assert obj.history == ""
        assert obj.use_count() == 1
        assert stash1.use_count(0) == 1
        stash1.Add(obj)
        assert obj.history == ""
        assert obj.use_count() == 2
        assert stash1.use_count(0) == 2
        assert stash1.use_count(1) == 2
        break  # Comment out for manual leak checking (use `top` command).


def test_release_to_cpp_stash_via_shared_from_this():
    obj = PySft("PySft")
    stash1 = m.SftSharedPtrStash(1)
    stash1.AddSharedFromThis(obj)
    assert obj.history == "PySft_Stash1AddSharedFromThis"
    assert stash1.use_count(0) == 2
    stash1.AddSharedFromThis(obj)
    assert obj.history == "PySft_Stash1AddSharedFromThis_Stash1AddSharedFromThis"
    assert stash1.use_count(0) == 3
    assert stash1.use_count(1) == 3


def test_release_to_cpp_stash_via_shared_from_this_leak_1():  # WIP
    m.to_cout("")
    m.to_cout("")
    m.to_cout("Add first")
    obj = PySft("")
    import weakref

    obj_wr = weakref.ref(obj)
    while True:
        stash1 = m.SftSharedPtrStash(1)
        stash1.Add(obj)
        assert obj.history == ""
        assert obj.use_count() == 1
        assert stash1.use_count(0) == 1
        stash1.AddSharedFromThis(obj)
        assert obj.history == ""
        assert obj.use_count() == 2
        assert stash1.use_count(0) == 2
        assert stash1.use_count(1) == 2
        del obj
        assert obj_wr() is not None
        assert stash1.use_count(0) == 2
        assert stash1.use_count(1) == 2
        break  # Comment out for manual leak checking (use `top` command).


def test_release_to_cpp_stash_via_shared_from_this_leak_2():  # WIP
    m.to_cout("")
    m.to_cout("AddSharedFromThis only")
    obj = PySft("")
    import weakref

    obj_wr = weakref.ref(obj)
    while True:
        stash1 = m.SftSharedPtrStash(1)
        stash1.AddSharedFromThis(obj)
        assert obj.history == ""
        assert obj.use_count() == 2
        assert stash1.use_count(0) == 2
        stash1.AddSharedFromThis(obj)
        assert obj.history == ""
        assert obj.use_count() == 3
        assert stash1.use_count(0) == 3
        assert stash1.use_count(1) == 3
        del obj
        assert obj_wr() is None  # BAD NEEDS FIXING
        assert stash1.use_count(0) == 2
        assert stash1.use_count(1) == 2
        break  # Comment out for manual leak checking (use `top` command).


def test_pass_released_shared_ptr_as_unique_ptr():
    obj = PySft("PySft")
    stash1 = m.SftSharedPtrStash(1)
    stash1.Add(obj)  # Releases shared_ptr to C++.
    with pytest.raises(ValueError) as exc_info:
        m.pass_unique_ptr(obj)
    assert str(exc_info.value) == (
        "Python instance is currently owned by a std::shared_ptr."
    )
