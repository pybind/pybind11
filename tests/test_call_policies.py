from __future__ import annotations

import pytest

import env  # noqa: F401
from pybind11_tests import ConstructorStats
from pybind11_tests import call_policies as m


@pytest.mark.xfail("env.PYPY", reason="sometimes comes out 1 off on PyPy", strict=False)
@pytest.mark.skipif("env.GRAALPY", reason="Cannot reliably trigger GC")
def test_keep_alive_argument(capture):
    n_inst = ConstructorStats.detail_reg_inst()
    with capture:
        p = m.Parent()
    assert capture == "Allocating parent."
    with capture:
        p.addChild(m.Child())
        assert ConstructorStats.detail_reg_inst() == n_inst + 1
    assert (
        capture
        == """
        Allocating child.
        Releasing child.
    """
    )
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == "Releasing parent."

    with capture:
        p = m.Parent()
    assert capture == "Allocating parent."
    with capture:
        p.addChildKeepAlive(m.Child())
        assert ConstructorStats.detail_reg_inst() == n_inst + 2
    assert capture == "Allocating child."
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert (
        capture
        == """
        Releasing parent.
        Releasing child.
    """
    )

    p = m.Parent()
    c = m.Child()
    assert ConstructorStats.detail_reg_inst() == n_inst + 2
    m.free_function(p, c)
    del c
    assert ConstructorStats.detail_reg_inst() == n_inst + 2
    del p
    assert ConstructorStats.detail_reg_inst() == n_inst

    with pytest.raises(RuntimeError) as excinfo:
        m.invalid_arg_index()
    assert str(excinfo.value) == "Could not activate keep_alive!"


@pytest.mark.skipif("env.GRAALPY", reason="Cannot reliably trigger GC")
def test_keep_alive_return_value(capture):
    n_inst = ConstructorStats.detail_reg_inst()
    with capture:
        p = m.Parent()
    assert capture == "Allocating parent."
    with capture:
        p.returnChild()
        assert ConstructorStats.detail_reg_inst() == n_inst + 1
    assert (
        capture
        == """
        Allocating child.
        Releasing child.
    """
    )
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == "Releasing parent."

    with capture:
        p = m.Parent()
    assert capture == "Allocating parent."
    with capture:
        p.returnChildKeepAlive()
        assert ConstructorStats.detail_reg_inst() == n_inst + 2
    assert capture == "Allocating child."
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert (
        capture
        == """
        Releasing parent.
        Releasing child.
    """
    )

    p = m.Parent()
    assert ConstructorStats.detail_reg_inst() == n_inst + 1
    with capture:
        m.Parent.staticFunction(p)
        assert ConstructorStats.detail_reg_inst() == n_inst + 2
    assert capture == "Allocating child."
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert (
        capture
        == """
        Releasing parent.
        Releasing child.
    """
    )


# https://foss.heptapod.net/pypy/pypy/-/issues/2447
@pytest.mark.xfail("env.PYPY", reason="_PyObject_GetDictPtr is unimplemented")
@pytest.mark.skipif("env.GRAALPY", reason="Cannot reliably trigger GC")
def test_alive_gc(capture):
    n_inst = ConstructorStats.detail_reg_inst()
    p = m.ParentGC()
    p.addChildKeepAlive(m.Child())
    assert ConstructorStats.detail_reg_inst() == n_inst + 2
    lst = [p]
    lst.append(lst)  # creates a circular reference
    with capture:
        del p, lst
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert (
        capture
        == """
        Releasing parent.
        Releasing child.
    """
    )


@pytest.mark.skipif("env.GRAALPY", reason="Cannot reliably trigger GC")
def test_alive_gc_derived(capture):
    class Derived(m.Parent):
        pass

    n_inst = ConstructorStats.detail_reg_inst()
    p = Derived()
    p.addChildKeepAlive(m.Child())
    assert ConstructorStats.detail_reg_inst() == n_inst + 2
    lst = [p]
    lst.append(lst)  # creates a circular reference
    with capture:
        del p, lst
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert (
        capture
        == """
        Releasing parent.
        Releasing child.
    """
    )


@pytest.mark.skipif("env.GRAALPY", reason="Cannot reliably trigger GC")
def test_alive_gc_multi_derived(capture):
    class Derived(m.Parent, m.Child):
        def __init__(self):
            m.Parent.__init__(self)
            m.Child.__init__(self)

    n_inst = ConstructorStats.detail_reg_inst()
    p = Derived()
    p.addChildKeepAlive(m.Child())
    # +3 rather than +2 because Derived corresponds to two registered instances
    assert ConstructorStats.detail_reg_inst() == n_inst + 3
    lst = [p]
    lst.append(lst)  # creates a circular reference
    with capture:
        del p, lst
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert (
        capture
        == """
        Releasing parent.
        Releasing child.
        Releasing child.
    """
    )


@pytest.mark.skipif("env.GRAALPY", reason="Cannot reliably trigger GC")
def test_return_none(capture):
    n_inst = ConstructorStats.detail_reg_inst()
    with capture:
        p = m.Parent()
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
        p = m.Parent()
    assert capture == "Allocating parent."
    with capture:
        p.returnNullChildKeepAliveParent()
        assert ConstructorStats.detail_reg_inst() == n_inst + 1
    assert capture == ""
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == "Releasing parent."


@pytest.mark.skipif("env.GRAALPY", reason="Cannot reliably trigger GC")
def test_keep_alive_constructor(capture):
    n_inst = ConstructorStats.detail_reg_inst()

    with capture:
        p = m.Parent(m.Child())
        assert ConstructorStats.detail_reg_inst() == n_inst + 2
    assert (
        capture
        == """
        Allocating child.
        Allocating parent.
    """
    )
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert (
        capture
        == """
        Releasing parent.
        Releasing child.
    """
    )


def test_call_guard():
    assert m.unguarded_call() == "unguarded"
    assert m.guarded_call() == "guarded"

    assert m.multiple_guards_correct_order() == "guarded & guarded"
    assert m.multiple_guards_wrong_order() == "unguarded & guarded"

    if hasattr(m, "with_gil"):
        assert m.with_gil() == "GIL held"
        assert m.without_gil() == "GIL released"


def test_keep_alive_failed_overload():
    """keep_alive on an overload that fails argument conversion must not fire."""
    obj = m.KeepAliveOverload()
    # Calling with a str rejects the first overload (and crashed before the fix).
    assert isinstance(m.keep_alive_overload(obj, 1), m.KeepAliveOverload)
    assert isinstance(m.keep_alive_overload(obj, "x"), m.KeepAliveOverload)
    assert isinstance(m.keep_alive_overload_reverse(obj, 1), m.KeepAliveOverload)
    assert isinstance(m.keep_alive_overload_reverse(obj, "x"), m.KeepAliveOverload)
    with pytest.raises(TypeError):
        m.keep_alive_single(obj, "x")


@pytest.mark.xfail("env.PYPY", reason="sometimes comes out 1 off on PyPy", strict=False)
@pytest.mark.skipif("env.GRAALPY", reason="Cannot reliably trigger GC")
@pytest.mark.parametrize(
    ("function", "value"),
    [
        (m.keep_alive_overload_args, "x"),
        (m.keep_alive_overload_args_converting, 1),
        (m.keep_alive_late_reference_failure, None),
    ],
)
def test_keep_alive_failed_overload_args(function, value):
    """An argument-to-argument keep_alive must not fire for a rejected overload either."""
    n_inst = ConstructorStats.detail_reg_inst()
    p, c = m.Parent(), m.Child()
    assert ConstructorStats.detail_reg_inst() == n_inst + 2
    # Rejected candidates must not retain c, including failures while extracting a reference.
    if value is None:
        with pytest.raises(TypeError):
            function(p, c, value)
    else:
        function(p, c, value)
    del c
    assert ConstructorStats.detail_reg_inst() == n_inst + 1
    # The successful overload still keeps its child alive.
    m.keep_alive_overload_args(p, m.Child(), 1)
    assert ConstructorStats.detail_reg_inst() == n_inst + 2
    del p
    assert ConstructorStats.detail_reg_inst() == n_inst


@pytest.mark.parametrize(
    "function",
    [m.keep_alive_unregistered_return, m.keep_alive_unregistered_return_reverse],
)
def test_keep_alive_failed_return_conversion(function):
    """A failed return-value conversion must raise its own error, not a keep_alive one."""
    with pytest.raises(
        TypeError, match="Unable to convert function return value"
    ) as excinfo:
        function(m.KeepAliveOverload())
    assert "Unregistered type" in str(excinfo.value.__cause__)


@pytest.mark.skipif("env.GRAALPY", reason="Cannot reliably trigger GC")
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("value", [1, "x"])
def test_keep_alive_overload_retention(reverse, value):
    n_inst = ConstructorStats.detail_reg_inst()
    obj = m.KeepAliveOverload()
    function = m.keep_alive_overload_reverse if reverse else m.keep_alive_overload
    result = function(obj, value)
    assert ConstructorStats.detail_reg_inst() == n_inst + 2
    if reverse:
        del result
    else:
        del obj
    assert ConstructorStats.detail_reg_inst() == n_inst + 2
    if reverse:
        del obj
    else:
        del result
    assert ConstructorStats.detail_reg_inst() == n_inst


@pytest.mark.skipif("env.GRAALPY", reason="Cannot reliably trigger GC")
def test_keep_alive_failed_constructor_overload():
    n_inst = ConstructorStats.detail_reg_inst()
    child = m.Child()
    parent = m.Parent(child, "x")
    del child
    assert ConstructorStats.detail_reg_inst() == n_inst + 1
    del parent
    assert ConstructorStats.detail_reg_inst() == n_inst
    parent = m.Parent(m.Child(), 1)
    assert ConstructorStats.detail_reg_inst() == n_inst + 2
    del parent
    assert ConstructorStats.detail_reg_inst() == n_inst


def test_call_policy_hooks():
    events = []
    with pytest.raises(TypeError):
        m.call_policy_hooks(events, object())
    assert events == []
    with pytest.raises(TypeError):
        m.call_policy_hooks_late_reference(events, None)
    assert events == []
    # The int requires conversion to double and therefore the second overload-resolution pass.
    for value in ("x", 1):
        m.call_policy_hooks(events, value)
        assert events == ["precall", "call", "postcall"]
        events.clear()
    with pytest.raises(RuntimeError, match="call failed"):
        m.call_policy_hooks_throw(events)
    assert events == ["precall", "call"]
    events.clear()
    with pytest.raises(TypeError, match="Unable to convert function return value"):
        m.call_policy_hooks_unregistered_return(events)
    assert events == ["precall", "call", "postcall:null"]
    if hasattr(m, "call_policy_hooks_without_gil"):
        events.clear()
        assert m.call_policy_hooks_without_gil(events) == "GIL released"
        assert events == ["precall", "postcall"]


@pytest.mark.skipif("env.GRAALPY", reason="Cannot reliably trigger GC")
def test_call_policy_postcall_error(capture):
    n_inst = ConstructorStats.detail_reg_inst()
    events = []
    with capture:
        with pytest.raises(RuntimeError, match="postcall failed"):
            m.call_policy_hooks_throw_postcall(events)
        pytest.gc_collect()
    assert events == ["precall", "call", "postcall"]
    assert capture == "Allocating child.\nReleasing child."
    assert ConstructorStats.detail_reg_inst() == n_inst


@pytest.mark.skipif("env.GRAALPY", reason="Cannot reliably trigger GC")
def test_keep_alive_error(capture):
    """A keep_alive error must not leave side effects or leak the return value."""
    n_inst = ConstructorStats.detail_reg_inst()
    c = m.Child()
    events = []
    # An int nurse cannot hold a weak reference, so keep_alive<1, 2> fails.
    with pytest.raises(TypeError, match="weak reference"):
        m.keep_alive_error_args(1, c, events)
    assert events == []
    del c
    pytest.gc_collect()
    with capture:
        with pytest.raises(TypeError, match="weak reference"):
            m.keep_alive_error_return(1)
        pytest.gc_collect()
    assert capture == "Allocating child.\nReleasing child."
    assert ConstructorStats.detail_reg_inst() == n_inst
