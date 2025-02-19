from __future__ import annotations

import sys
import weakref

import pytest

import env
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
        assert not obj.history
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
        assert not obj.history
        assert m.use_count(obj) == 2
        assert stash1.use_count(0) == 1
        stash1.Add(obj)
        assert not obj.history
        assert m.use_count(obj) == 3
        assert stash1.use_count(0) == 2
        assert stash1.use_count(1) == 2
        break  # Comment out for manual leak checking (use `top` command).


def test_release_and_stash_via_shared_from_this():
    # Exercises that the smart_holder vptr is invisible to the shared_from_this mechanism.
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
        assert not obj.history
        assert stash1.use_count(0) == 1
        stash1.AddSharedFromThis(obj)
        assert not obj.history
        assert stash1.use_count(0) == 2
        assert stash1.use_count(1) == 2
        break  # Comment out for manual leak checking (use `top` command).


def test_pass_released_shared_ptr_as_unique_ptr():
    # Exercises that returning a unique_ptr fails while a shared_from_this
    # visible shared_ptr exists.
    obj = PySft("PySft")
    stash1 = m.SftSharedPtrStash(1)
    stash1.Add(obj)  # Releases shared_ptr to C++.
    assert m.pass_unique_ptr_cref(obj) == "PySft_Stash1Add"
    assert obj.history == "PySft_Stash1Add"
    with pytest.raises(ValueError) as exc_info:
        m.pass_unique_ptr_rref(obj)
    assert str(exc_info.value) == (
        "Python instance is currently owned by a std::shared_ptr."
    )
    assert obj.history == "PySft_Stash1Add"


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


def test_multiple_registered_instances_for_same_pointee():
    obj0 = PySft("PySft")
    obj0.attachment_in_dict = "Obj0"
    assert m.pass_through_shd_ptr(obj0) is obj0
    while True:
        obj = m.Sft(obj0)
        assert obj is not obj0
        obj_pt = m.pass_through_shd_ptr(obj)
        # Unpredictable! Because registered_instances is as std::unordered_multimap.
        assert obj_pt is obj0 or obj_pt is obj
        # Multiple registered_instances for the same pointee can lead to unpredictable results:
        if obj_pt is obj0:
            assert obj_pt.attachment_in_dict == "Obj0"
        else:
            assert not hasattr(obj_pt, "attachment_in_dict")
        assert obj0.history == "PySft"
        break  # Comment out for manual leak checking (use `top` command).


def test_multiple_registered_instances_for_same_pointee_leak():
    obj0 = PySft("")
    while True:
        stash1 = m.SftSharedPtrStash(1)
        stash1.Add(m.Sft(obj0))
        assert stash1.use_count(0) == 1
        stash1.Add(m.Sft(obj0))
        assert stash1.use_count(0) == 1
        assert stash1.use_count(1) == 1
        assert not obj0.history
        break  # Comment out for manual leak checking (use `top` command).


def test_multiple_registered_instances_for_same_pointee_recursive():
    while True:
        obj0 = PySft("PySft")
        if not env.PYPY:
            obj0_wr = weakref.ref(obj0)
        obj = obj0
        # This loop creates a chain of instances linked by shared_ptrs.
        for _ in range(10):
            obj_next = m.Sft(obj)
            assert obj_next is not obj
            obj = obj_next
            del obj_next
            assert obj.history == "PySft"
        del obj0
        if not env.PYPY and not env.GRAALPY:
            assert obj0_wr() is not None
        del obj  # This releases the chain recursively.
        if not env.PYPY and not env.GRAALPY:
            assert obj0_wr() is None
        break  # Comment out for manual leak checking (use `top` command).


# As of 2021-07-10 the pybind11 GitHub Actions valgrind build uses Python 3.9.
WORKAROUND_ENABLING_ROLLBACK_OF_PR3068 = env.LINUX and sys.version_info == (3, 9)


def test_std_make_shared_factory():
    class PySftMakeShared(m.Sft):
        def __init__(self, history):
            super().__init__(history, 0)

    obj = PySftMakeShared("PySftMakeShared")
    assert obj.history == "PySftMakeShared"
    if WORKAROUND_ENABLING_ROLLBACK_OF_PR3068:
        try:
            m.pass_through_shd_ptr(obj)
        except RuntimeError as e:
            str_exc_info_value = str(e)
        else:
            str_exc_info_value = "RuntimeError NOT RAISED"
    else:
        with pytest.raises(RuntimeError) as exc_info:
            m.pass_through_shd_ptr(obj)
        str_exc_info_value = str(exc_info.value)
    assert (
        str_exc_info_value
        == "smart_holder_type_casters load_as_shared_ptr failure: not implemented:"
        " trampoline-self-life-support for external shared_ptr to type inheriting"
        " from std::enable_shared_from_this."
    )
