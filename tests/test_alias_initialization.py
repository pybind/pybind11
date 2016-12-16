import pytest


def test_alias_delay_initialization1(capture):
    """
    A only initializes its trampoline class when we inherit from it; if we just
    create and use an A instance directly, the trampoline initialization is
    bypassed and we only initialize an A() instead (for performance reasons).
    """
    from pybind11_tests import A, call_f

    class B(A):
        def __init__(self):
            super(B, self).__init__()

        def f(self):
            print("In python f()")

    # C++ version
    with capture:
        a = A()
        call_f(a)
        del a
        pytest.gc_collect()
    assert capture == "A.f()"

    # Python version
    with capture:
        b = B()
        call_f(b)
        del b
        pytest.gc_collect()
    assert capture == """
        PyA.PyA()
        PyA.f()
        In python f()
        PyA.~PyA()
    """


def test_alias_delay_initialization2(capture):
    """A2, unlike the above, is configured to always initialize the alias; while
    the extra initialization and extra class layer has small virtual dispatch
    performance penalty, it also allows us to do more things with the trampoline
    class such as defining local variables and performing construction/destruction.
    """
    from pybind11_tests import A2, call_f

    class B2(A2):
        def __init__(self):
            super(B2, self).__init__()

        def f(self):
            print("In python B2.f()")

    # No python subclass version
    with capture:
        a2 = A2()
        call_f(a2)
        del a2
        pytest.gc_collect()
    assert capture == """
        PyA2.PyA2()
        PyA2.f()
        A2.f()
        PyA2.~PyA2()
    """

    # Python subclass version
    with capture:
        b2 = B2()
        call_f(b2)
        del b2
        pytest.gc_collect()
    assert capture == """
        PyA2.PyA2()
        PyA2.f()
        In python B2.f()
        PyA2.~PyA2()
    """
