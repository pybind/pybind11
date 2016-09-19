import pytest
from pybind11_tests import (kw_func0, kw_func1, kw_func2, kw_func3, kw_func4, args_function,
                            args_kwargs_function, kw_func_udl, kw_func_udl_z, KWClass)


def test_function_signatures(doc):
    assert doc(kw_func0) == "kw_func0(arg0: int, arg1: int) -> str"
    assert doc(kw_func1) == "kw_func1(x: int, y: int) -> str"
    assert doc(kw_func2) == "kw_func2(x: int=100, y: int=200) -> str"
    assert doc(kw_func3) == "kw_func3(data: str='Hello world!') -> None"
    assert doc(kw_func4) == "kw_func4(myList: List[int]=[13, 17]) -> str"
    assert doc(kw_func_udl) == "kw_func_udl(x: int, y: int=300) -> str"
    assert doc(kw_func_udl_z) == "kw_func_udl_z(x: int, y: int=0) -> str"
    assert doc(args_function) == "args_function(*args) -> tuple"
    assert doc(args_kwargs_function) == "args_kwargs_function(*args, **kwargs) -> tuple"
    assert doc(KWClass.foo0) == "foo0(self: m.KWClass, arg0: int, arg1: float) -> None"
    assert doc(KWClass.foo1) == "foo1(self: m.KWClass, x: int, y: float) -> None"


def test_named_arguments(msg):
    assert kw_func0(5, 10) == "x=5, y=10"

    assert kw_func1(5, 10) == "x=5, y=10"
    assert kw_func1(5, y=10) == "x=5, y=10"
    assert kw_func1(y=10, x=5) == "x=5, y=10"

    assert kw_func2() == "x=100, y=200"
    assert kw_func2(5) == "x=5, y=200"
    assert kw_func2(x=5) == "x=5, y=200"
    assert kw_func2(y=10) == "x=100, y=10"
    assert kw_func2(5, 10) == "x=5, y=10"
    assert kw_func2(x=5, y=10) == "x=5, y=10"

    with pytest.raises(TypeError) as excinfo:
        # noinspection PyArgumentList
        kw_func2(x=5, y=10, z=12)
    assert msg(excinfo.value) == """
        kw_func2(): incompatible function arguments. The following argument types are supported:
            1. (x: int=100, y: int=200) -> str

        Invoked with:
    """

    assert kw_func4() == "{13 17}"
    assert kw_func4(myList=[1, 2, 3]) == "{1 2 3}"

    assert kw_func_udl(x=5, y=10) == "x=5, y=10"
    assert kw_func_udl_z(x=5) == "x=5, y=0"


def test_arg_and_kwargs():
    args = 'arg1_value', 'arg2_value', 3
    assert args_function(*args) == args

    args = 'a1', 'a2'
    kwargs = dict(arg3='a3', arg4=4)
    assert args_kwargs_function(*args, **kwargs) == (args, kwargs)
