import gc


def test_keep_alive_argument(capture):
    from pybind11_tests import Parent, Child

    with capture:
        p = Parent()
    assert capture == "Allocating parent."
    with capture:
        p.addChild(Child())
        gc.collect()
    assert capture == """
        Allocating child.
        Releasing child.
    """
    with capture:
        del p
        gc.collect()
    assert capture == "Releasing parent."

    with capture:
        p = Parent()
    assert capture == "Allocating parent."
    with capture:
        p.addChildKeepAlive(Child())
        gc.collect()
    assert capture == "Allocating child."
    with capture:
        del p
        gc.collect()
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
        gc.collect()
    assert capture == """
        Allocating child.
        Releasing child.
    """
    with capture:
        del p
        gc.collect()
    assert capture == "Releasing parent."

    with capture:
        p = Parent()
    assert capture == "Allocating parent."
    with capture:
        p.returnChildKeepAlive()
        gc.collect()
    assert capture == "Allocating child."
    with capture:
        del p
        gc.collect()
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
        gc.collect()
    assert capture == ""
    with capture:
        del p
        gc.collect()
    assert capture == "Releasing parent."

    with capture:
        p = Parent()
    assert capture == "Allocating parent."
    with capture:
        p.returnNullChildKeepAliveParent()
        gc.collect()
    assert capture == ""
    with capture:
        del p
        gc.collect()
    assert capture == "Releasing parent."
