# -*- coding: utf-8 -*-
from pybind11_tests import docstring_function_signature as m
import sys


def test_docstring_function_signature():
    def syntactically_valid(sig):
        try:
            complete_fnsig = "def " + sig + ": pass"
            ast.parse(complete_fnsig)
            return True
        except SyntaxError:
            return False

    pass

    methods = ["a", "b", "c", "d", "e", "f", "g"]
    root_module = "pybind11_tests"
    module = "{}.{}".format(root_module, "docstring_function_signature")
    expected_signatures = [
        "a(a: {0}.Color = {0}.Color.Red) -> None".format(module),
        "b(a: int = 1) -> None",
        "c(a: List[int] = [1, 2, 3, 4]) -> None",
        "d(a: {}.UserType = ...) -> None".format(root_module),
        "e(a: Tuple[{}.UserType, int] = (..., 4)) -> None".format(root_module),
        "f(a: List[{0}.Color] = [{0}.Color.Red]) -> None".format(module),
        "g(a: Tuple[int, {0}.Color, float] = (4, {0}.Color.Red, 1.9)) -> None".format(
            module
        ),
    ]

    for method, signature in zip(methods, expected_signatures):
        docstring = getattr(m, method).__doc__.strip("\n")
        assert docstring == signature

    if sys.version_info.major >= 3 and sys.version_info.minor >= 5:
        import ast

        for method in methods:
            docstring = getattr(m, method).__doc__.strip("\n")
            assert syntactically_valid(docstring)
