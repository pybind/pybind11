import pytest


def test_keep_alive_argument(capture):
    from pybind11_tests import Parent, Child, ConstructorStats

    n_inst = ConstructorStats.detail_reg_inst()
    with capture:
        p = Parent()
    assert capture == "Allocating parent."
    with capture:
        p.addChild(Child())
        assert ConstructorStats.detail_reg_inst() == n_inst + 1
    assert capture == """
        Allocating child.
        Releasing child.
    """
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == "Releasing parent."

    with capture:
        p = Parent()
    assert capture == "Allocating parent."
    with capture:
        p.addChildKeepAlive(Child())
        assert ConstructorStats.detail_reg_inst() == n_inst + 2
    assert capture == "Allocating child."
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == """
        Releasing parent.
        Releasing child.
    """


def test_keep_alive_return_value(capture):
    from pybind11_tests import Parent, ConstructorStats

    n_inst = ConstructorStats.detail_reg_inst()
    with capture:
        p = Parent()
    assert capture == "Allocating parent."
    with capture:
        p.returnChild()
        assert ConstructorStats.detail_reg_inst() == n_inst + 1
    assert capture == """
        Allocating child.
        Releasing child.
    """
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == "Releasing parent."

    with capture:
        p = Parent()
    assert capture == "Allocating parent."
    with capture:
        p.returnChildKeepAlive()
        assert ConstructorStats.detail_reg_inst() == n_inst + 2
    assert capture == "Allocating child."
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == """
        Releasing parent.
        Releasing child.
    """


# https://bitbucket.org/pypy/pypy/issues/2447
@pytest.unsupported_on_pypy
def test_alive_gc(capture):
    from pybind11_tests import ParentGC, Child, ConstructorStats

    n_inst = ConstructorStats.detail_reg_inst()
    p = ParentGC()
    p.addChildKeepAlive(Child())
    assert ConstructorStats.detail_reg_inst() == n_inst + 2
    lst = [p]
    lst.append(lst)   # creates a circular reference
    with capture:
        del p, lst
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == """
        Releasing parent.
        Releasing child.
    """


def test_alive_gc_derived(capture):
    from pybind11_tests import Parent, Child, ConstructorStats

    class Derived(Parent):
        pass

    n_inst = ConstructorStats.detail_reg_inst()
    p = Derived()
    p.addChildKeepAlive(Child())
    assert ConstructorStats.detail_reg_inst() == n_inst + 2
    lst = [p]
    lst.append(lst)   # creates a circular reference
    with capture:
        del p, lst
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == """
        Releasing parent.
        Releasing child.
    """


def test_alive_gc_multi_derived(capture):
    from pybind11_tests import Parent, Child, ConstructorStats

    class Derived(Parent, Child):
        pass

    n_inst = ConstructorStats.detail_reg_inst()
    p = Derived()
    p.addChildKeepAlive(Child())
    # +3 rather than +2 because Derived corresponds to two registered instances
    assert ConstructorStats.detail_reg_inst() == n_inst + 3
    lst = [p]
    lst.append(lst)   # creates a circular reference
    with capture:
        del p, lst
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == """
        Releasing parent.
        Releasing child.
    """


def test_return_none(capture):
    from pybind11_tests import Parent, ConstructorStats

    n_inst = ConstructorStats.detail_reg_inst()
    with capture:
        p = Parent()
    assert capture == "Allocating parent."
    with capture:
        p.returnNullChildKeepAliveChild()
        assert ConstructorStats.detail_reg_inst() == n_inst + 1
    assert capture == ""
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == "Releasing parent."

    with capture:
        p = Parent()
    assert capture == "Allocating parent."
    with capture:
        p.returnNullChildKeepAliveParent()
        assert ConstructorStats.detail_reg_inst() == n_inst + 1
    assert capture == ""
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == "Releasing parent."


def test_call_guard():
    from pybind11_tests import call_policies

    assert call_policies.unguarded_call() == "unguarded"
    assert call_policies.guarded_call() == "guarded"

    assert call_policies.multiple_guards_correct_order() == "guarded & guarded"
    assert call_policies.multiple_guards_wrong_order() == "unguarded & guarded"

    if hasattr(call_policies, "with_gil"):
        assert call_policies.with_gil() == "GIL held"
        assert call_policies.without_gil() == "GIL released"
