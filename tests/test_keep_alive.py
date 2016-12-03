import gc


def collect():
    gc.collect()
    gc.collect()


def test_keep_alive_argument(capture):
    from pybind11_tests import Parent, Child

    with capture:
        p = Parent()
    assert capture == "Allocating parent."
    with capture:
        p.addChild(Child())
        collect()
    assert capture == """
        Allocating child.
        Releasing child.
    """
    with capture:
        del p
        collect()
    assert capture == "Releasing parent."

    with capture:
        p = Parent()
    assert capture == "Allocating parent."
    with capture:
        p.addChildKeepAlive(Child())
        collect()
    assert capture == "Allocating child."
    with capture:
        del p
        collect()
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
        collect()
    assert capture == """
        Allocating child.
        Releasing child.
    """
    with capture:
        del p
        collect()
    assert capture == "Releasing parent."

    with capture:
        p = Parent()
    assert capture == "Allocating parent."
    with capture:
        p.returnChildKeepAlive()
        collect()
    assert capture == "Allocating child."
    with capture:
        del p
        collect()
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
        collect()
    assert capture == ""
    with capture:
        del p
        collect()
    assert capture == "Releasing parent."

    with capture:
        p = Parent()
    assert capture == "Allocating parent."
    with capture:
        p.returnNullChildKeepAliveParent()
        collect()
    assert capture == ""
    with capture:
        del p
        collect()
    assert capture == "Releasing parent."
