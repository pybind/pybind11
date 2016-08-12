import pytest
import gc


def test_regressions(capture):
    from pybind11_tests.issues import print_cchar, print_char

    with capture:
        print_cchar("const char *")  # #137: const char* isn't handled properly
    assert capture == "const char *"
    with capture:
        print_char("c")  # #150: char bindings broken
    assert capture == "c"


def test_dispatch_issue(capture, msg):
    """#159: virtual function dispatch has problems with similar-named functions"""
    from pybind11_tests.issues import DispatchIssue, dispatch_issue_go

    class PyClass1(DispatchIssue):
        def dispatch(self):
            print("Yay..")

    class PyClass2(DispatchIssue):
        def dispatch(self):
            with pytest.raises(RuntimeError) as excinfo:
                super(PyClass2, self).dispatch()
            assert msg(excinfo.value) == 'Tried to call pure virtual function "Base::dispatch"'

            p = PyClass1()
            dispatch_issue_go(p)

    b = PyClass2()
    with capture:
        dispatch_issue_go(b)
    assert capture == "Yay.."


def test_reference_wrapper():
    """#171: Can't return reference wrappers (or STL data structures containing them)"""
    from pybind11_tests.issues import Placeholder, return_vec_of_reference_wrapper

    assert str(return_vec_of_reference_wrapper(Placeholder(4))) == \
        "[Placeholder[1], Placeholder[2], Placeholder[3], Placeholder[4]]"


def test_iterator_passthrough():
    """#181: iterator passthrough did not compile"""
    from pybind11_tests.issues import iterator_passthrough

    assert list(iterator_passthrough(iter([3, 5, 7, 9, 11, 13, 15]))) == [3, 5, 7, 9, 11, 13, 15]


def test_shared_ptr_gc():
    """// #187: issue involving std::shared_ptr<> return value policy & garbage collection"""
    from pybind11_tests.issues import ElementList, ElementA

    el = ElementList()
    for i in range(10):
        el.add(ElementA(i))
    gc.collect()
    for i, v in enumerate(el.get()):
        assert i == v.value()


def test_no_id(capture, msg):
    from pybind11_tests.issues import print_element, expect_float, expect_int

    with pytest.raises(TypeError) as excinfo:
        print_element(None)
    assert msg(excinfo.value) == """
        Incompatible function arguments. The following argument types are supported:
            1. (arg0: m.issues.ElementA) -> None
            Invoked with: None
    """

    with pytest.raises(TypeError) as excinfo:
        expect_int(5.2)
    assert msg(excinfo.value) == """
        Incompatible function arguments. The following argument types are supported:
            1. (arg0: int) -> int
            Invoked with: 5.2
    """
    assert expect_float(12) == 12

    from pybind11_tests.issues import A, call_f

    class B(A):
        def __init__(self):
            super(B, self).__init__()

        def f(self):
            print("In python f()")

    # C++ version
    with capture:
        a = A()
        call_f(a)
    assert capture == "A.f()"

    # Python version
    with capture:
        b = B()
        call_f(b)
    assert capture == """
        PyA.PyA()
        PyA.f()
        In python f()
    """


def test_str_issue(capture, msg):
    """Issue #283: __str__ called on uninitialized instance when constructor arguments invalid"""
    from pybind11_tests.issues import StrIssue

    with capture:
        assert str(StrIssue(3)) == "StrIssue[3]"
    assert capture == "StrIssue.__str__ called"

    with pytest.raises(TypeError) as excinfo:
        str(StrIssue("no", "such", "constructor"))
    assert msg(excinfo.value) == """
        Incompatible constructor arguments. The following argument types are supported:
            1. m.issues.StrIssue(arg0: int)
            2. m.issues.StrIssue()
            Invoked with: no, such, constructor
    """


def test_nested(capture):
    """ #328: first member in a class can't be used in operators"""
    from pybind11_tests.issues import NestA, NestB, NestC, print_NestA, print_NestB, print_NestC

    a = NestA()
    b = NestB()
    c = NestC()

    a += 10
    b.a += 100
    c.b.a += 1000
    b -= 1
    c.b -= 3
    c *= 7

    with capture:
        print_NestA(a)
        print_NestA(b.a)
        print_NestA(c.b.a)
        print_NestB(b)
        print_NestB(c.b)
        print_NestC(c)
    assert capture == """
        13
        103
        1003
        3
        1
        35
    """

    abase = a.as_base()
    assert abase.value == -2
    a.as_base().value += 44
    assert abase.value == 42
    assert c.b.a.as_base().value == -2
    c.b.a.as_base().value += 44
    assert c.b.a.as_base().value == 42

    del c
    gc.collect()
    del a  # Should't delete while abase is still alive
    gc.collect()

    assert abase.value == 42
    del abase, b
    gc.collect()
