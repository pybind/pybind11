from pybind11_tests import builtins as m


def test_function_callable():
    def func():
        pass
    assert m.is_callable(func)


def test_class_callable():
    class C:
        pass
    assert m.is_callable(C)


def test_str_not_callable():
    s = "test"
    assert not m.is_callable(s)


def test_obj_not_callable():
    class C:
        pass

    c = C()
    assert not m.is_callable(c)


def test_obj_callable():
    class C:
        def __call__():
            pass

    c = C()
    assert m.is_callable(c)
