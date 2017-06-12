import pytest


def test_keep_alive_argument(capture):
    from pybind11_tests import Parent, Child

    with capture:
        p = Parent()
    assert capture == "Allocating parent."
    with capture:
        p.addChild(Child())
        pytest.gc_collect()
    assert capture == """
        Allocating child.
        Releasing child.
    """
    with capture:
        del p
        pytest.gc_collect()
    assert capture == "Releasing parent."

    with capture:
        p = Parent()
    assert capture == "Allocating parent."
    with capture:
        p.addChildKeepAlive(Child())
        pytest.gc_collect()
    assert capture == "Allocating child."
    with capture:
        del p
        pytest.gc_collect()
    assert capture == """
        Releasing parent.
        Releasing child.
    """


def test_keep_alive_return_value(capture):
    from pybind11_tests import Parent

    with capture:
        p = Parent()
    assert capture == "Allocating parent."
    with capture:
        p.returnChild()
        pytest.gc_collect()
    assert capture == """
        Allocating child.
        Releasing child.
    """
    with capture:
        del p
        pytest.gc_collect()
    assert capture == "Releasing parent."

    with capture:
        p = Parent()
    assert capture == "Allocating parent."
    with capture:
        p.returnChildKeepAlive()
        pytest.gc_collect()
    assert capture == "Allocating child."
    with capture:
        del p
        pytest.gc_collect()
    assert capture == """
        Releasing parent.
        Releasing child.
    """


def test_return_none(capture):
    from pybind11_tests import Parent

    with capture:
        p = Parent()
    assert capture == "Allocating parent."
    with capture:
        p.returnNullChildKeepAliveChild()
        pytest.gc_collect()
    assert capture == ""
    with capture:
        del p
        pytest.gc_collect()
    assert capture == "Releasing parent."

    with capture:
        p = Parent()
    assert capture == "Allocating parent."
    with capture:
        p.returnNullChildKeepAliveParent()
        pytest.gc_collect()
    assert capture == ""
    with capture:
        del p
        pytest.gc_collect()
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
