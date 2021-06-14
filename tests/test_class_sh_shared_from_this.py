# -*- coding: utf-8 -*-
import pytest

from pybind11_tests import class_sh_shared_from_this as m
from pybind11_tests import ConstructorStats


def test_smart_ptr(capture):
    # Object3
    for i, o in zip(
        [9, 8, 9], [m.MyObject3(9), m.make_myobject3_1(), m.make_myobject3_2()]
    ):
        print(o)
        with capture:
            m.print_myobject3_1(o)
            m.print_myobject3_2(o)
            m.print_myobject3_3(o)
            m.print_myobject3_3(o)  # XXX XXX XXX print_myobject3_4
        assert capture == "MyObject3[{i}]\n".format(i=i) * 4

    cstats = ConstructorStats.get(m.MyObject3)
    assert cstats.alive() == 1
    o = None
    assert cstats.alive() == 0
    assert cstats.values() == ["MyObject3[9]", "MyObject3[8]", "MyObject3[9]"]
    assert cstats.default_constructions == 0
    assert cstats.copy_constructions == 0
    # assert cstats.move_constructions >= 0 # Doesn't invoke any
    assert cstats.copy_assignments == 0
    assert cstats.move_assignments == 0


def test_shared_from_this_ref():
    s = m.SharedFromThisRef()
    stats = ConstructorStats.get(m.B)
    assert stats.alive() == 2

    ref = s.ref  # init_holder_helper(holder_ptr=false, owned=false, bad_wp=false)
    assert stats.alive() == 2
    assert s.set_ref(ref)
    assert s.set_holder(
        ref
    )  # std::enable_shared_from_this can create a holder from a reference
    del ref, s
    assert stats.alive() == 0


def test_shared_from_this_bad_wp():
    s = m.SharedFromThisRef()
    stats = ConstructorStats.get(m.B)
    assert stats.alive() == 2

    bad_wp = s.bad_wp  # init_holder_helper(holder_ptr=false, owned=false, bad_wp=true)
    assert stats.alive() == 2
    assert s.set_ref(bad_wp)
    with pytest.raises(RuntimeError) as excinfo:
        assert s.set_holder(bad_wp)
    assert str(excinfo.value) == "Non-owning holder (loaded_as_shared_ptr)."
    del bad_wp, s
    assert stats.alive() == 0


def test_shared_from_this_copy():
    s = m.SharedFromThisRef()
    stats = ConstructorStats.get(m.B)
    assert stats.alive() == 2

    copy = s.copy  # init_holder_helper(holder_ptr=false, owned=true, bad_wp=false)
    # RuntimeError: Invalid return_value_policy for shared_ptr.
    assert stats.alive() == 3
    assert s.set_ref(copy)
    assert s.set_holder(copy)
    del copy, s
    assert stats.alive() == 0


def test_shared_from_this_holder_ref():
    s = m.SharedFromThisRef()
    stats = ConstructorStats.get(m.B)
    assert stats.alive() == 2

    holder_ref = (
        s.holder_ref
    )  # init_holder_helper(holder_ptr=true, owned=false, bad_wp=false)
    assert stats.alive() == 2
    assert s.set_ref(holder_ref)
    assert s.set_holder(holder_ref)
    del holder_ref, s
    assert stats.alive() == 0


def test_shared_from_this_holder_copy():
    s = m.SharedFromThisRef()
    stats = ConstructorStats.get(m.B)
    assert stats.alive() == 2

    holder_copy = (
        # RuntimeError: Invalid return_value_policy for shared_ptr.
        s.holder_copy
    )  # init_holder_helper(holder_ptr=true, owned=true, bad_wp=false)
    assert stats.alive() == 2
    assert s.set_ref(holder_copy)
    assert s.set_holder(holder_copy)
    del holder_copy, s
    assert stats.alive() == 0


def test_shared_from_this_virt():
    z = m.SharedFromThisVirt.get()
    y = m.SharedFromThisVirt.get()
    assert y is z


@pytest.mark.parametrize(
    "test_func",
    [
        m.obj1_owns,
        m.obj2_owns,
        m.obj_sft_reset,
    ],
)
def test_shared_from_this_custom_deleters(test_func):
    test_func()


def test_shared_ptr_reset_and_rescue_pointee_model(loop_count_max=10 ** 6):
    loop_count = 0
    while True:
        m.shared_ptr_reset_and_rescue_pointee_model_proof_of_concept()
        loop_count += 1
        if loop_count == loop_count_max:
            break
