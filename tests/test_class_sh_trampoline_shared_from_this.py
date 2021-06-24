# -*- coding: utf-8 -*-
import pytest

import pybind11_tests.class_sh_trampoline_shared_from_this as m


class PySft(m.Sft):
    pass


def test_release_and_shared_from_this():
    # Exercises the most direct path from building a shared_from_this-visible
    # shared_ptr to calling shared_from_this.
    obj = PySft("PySft")
    assert obj.history == "PySft"
    assert m.use_count(obj) == 1
    assert m.pass_shared_ptr(obj) == 2
    assert obj.history == "PySft_PassSharedPtr"
    assert m.use_count(obj) == 1
    assert m.pass_shared_ptr(obj) == 2
    assert obj.history == "PySft_PassSharedPtr_PassSharedPtr"
    assert m.use_count(obj) == 1


def test_release_and_shared_from_this_leak():
    obj = PySft("")
    while True:
        m.pass_shared_ptr(obj)
        assert obj.history == ""
        assert m.use_count(obj) == 1
        break  # Comment out for manual leak checking (use `top` command).


def test_release_and_stash():
    # Exercises correct functioning of guarded_delete weak_ptr.
    obj = PySft("PySft")
    stash1 = m.SftSharedPtrStash(1)
    stash1.Add(obj)
    exp_hist = "PySft_Stash1Add"
    assert obj.history == exp_hist
    assert m.use_count(obj) == 2
    assert stash1.history(0) == exp_hist
    assert stash1.use_count(0) == 1
    assert m.pass_shared_ptr(obj) == 3
    exp_hist += "_PassSharedPtr"
    assert obj.history == exp_hist
    assert m.use_count(obj) == 2
    assert stash1.history(0) == exp_hist
    assert stash1.use_count(0) == 1
    stash2 = m.SftSharedPtrStash(2)
    stash2.Add(obj)
    exp_hist += "_Stash2Add"
    assert obj.history == exp_hist
    assert m.use_count(obj) == 3
    assert stash2.history(0) == exp_hist
    assert stash2.use_count(0) == 2
    stash2.Add(obj)
    exp_hist += "_Stash2Add"
    assert obj.history == exp_hist
    assert m.use_count(obj) == 4
    assert stash1.history(0) == exp_hist
    assert stash1.use_count(0) == 3
    assert stash2.history(0) == exp_hist
    assert stash2.use_count(0) == 3
    assert stash2.history(1) == exp_hist
    assert stash2.use_count(1) == 3
    del obj
    assert stash2.history(0) == exp_hist
    assert stash2.use_count(0) == 3
    assert stash2.history(1) == exp_hist
    assert stash2.use_count(1) == 3
    stash2.Clear()
    assert stash1.history(0) == exp_hist
    assert stash1.use_count(0) == 1


def test_release_and_stash_leak():
    obj = PySft("")
    while True:
        stash1 = m.SftSharedPtrStash(1)
        stash1.Add(obj)
        assert obj.history == ""
        assert m.use_count(obj) == 2
        assert stash1.use_count(0) == 1
        stash1.Add(obj)
        assert obj.history == ""
        assert m.use_count(obj) == 3
        assert stash1.use_count(0) == 2
        assert stash1.use_count(1) == 2
        break  # Comment out for manual leak checking (use `top` command).


def test_release_and_stash_via_shared_from_this():
    # Exercises that the smart_holder vptr is invisible to the shared_from_this mechnism.
    obj = PySft("PySft")
    stash1 = m.SftSharedPtrStash(1)
    with pytest.raises(RuntimeError) as exc_info:
        stash1.AddSharedFromThis(obj)
    assert str(exc_info.value) == "bad_weak_ptr"
    stash1.Add(obj)
    assert obj.history == "PySft_Stash1Add"
    assert stash1.use_count(0) == 1
    stash1.AddSharedFromThis(obj)
    assert obj.history == "PySft_Stash1Add_Stash1AddSharedFromThis"
    assert stash1.use_count(0) == 2
    assert stash1.use_count(1) == 2


def test_release_and_stash_via_shared_from_this_leak():
    obj = PySft("")
    while True:
        stash1 = m.SftSharedPtrStash(1)
        with pytest.raises(RuntimeError) as exc_info:
            stash1.AddSharedFromThis(obj)
        assert str(exc_info.value) == "bad_weak_ptr"
        stash1.Add(obj)
        assert obj.history == ""
        assert stash1.use_count(0) == 1
        stash1.AddSharedFromThis(obj)
        assert obj.history == ""
        assert stash1.use_count(0) == 2
        assert stash1.use_count(1) == 2
        break  # Comment out for manual leak checking (use `top` command).


def test_pass_released_shared_ptr_as_unique_ptr():
    # Exercises that returning a unique_ptr fails while a shared_from_this
    # visible shared_ptr exists.
    obj = PySft("PySft")
    stash1 = m.SftSharedPtrStash(1)
    stash1.Add(obj)  # Releases shared_ptr to C++.
    with pytest.raises(ValueError) as exc_info:
        m.pass_unique_ptr(obj)
    assert str(exc_info.value) == (
        "Python instance is currently owned by a std::shared_ptr."
    )


@pytest.mark.parametrize(
    "make_f",
    [
        m.make_pure_cpp_sft_raw_ptr,
        m.make_pure_cpp_sft_unq_ptr,
        m.make_pure_cpp_sft_shd_ptr,
    ],
)
def test_pure_cpp_sft_raw_ptr(make_f):
    # Exercises void_cast_raw_ptr logic for different situations.
    obj = make_f("PureCppSft")
    assert m.pass_shared_ptr(obj) == 3
    assert obj.history == "PureCppSft_PassSharedPtr"
    obj = make_f("PureCppSft")
    stash1 = m.SftSharedPtrStash(1)
    stash1.AddSharedFromThis(obj)
    assert obj.history == "PureCppSft_Stash1AddSharedFromThis"
