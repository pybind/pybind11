import itertools

import pytest

import pybind11_cross_module_tests
import pybind11_tests

m = pytest.importorskip("pybind11_tests.constants_and_functions")


def test_namespace_visibility():
    mdls = (
        pybind11_tests,
        pybind11_tests.constants_and_functions,
        pybind11_cross_module_tests,
    )
    codes = []
    for vis in itertools.product(*([("u", "h")] * len(mdls))):
        func = "ns_vis_" + "".join(vis) + "_func"
        addrs = []
        code = ""
        for v, mdl in zip(vis, mdls):
            addr = getattr(mdl, func)(True)
            addrs.append(addr)
            c = "ABC"[addrs.index(addr)]
            if v == "h":
                c = c.lower()
            code += c
        codes.append(code)
    code_line = ":".join(codes)
    if code_line != "AAC:AAc:AaC:Aac:aAC:aAc:aaC:aac":
        pytest.skip(f"UNEXPECTED code_line: {code_line}")


def test_constants():
    assert m.some_constant == 14


def test_function_overloading():
    assert m.test_function() == "test_function()"
    assert m.test_function(7) == "test_function(7)"
    assert m.test_function(m.MyEnum.EFirstEntry) == "test_function(enum=1)"
    assert m.test_function(m.MyEnum.ESecondEntry) == "test_function(enum=2)"

    assert m.test_function() == "test_function()"
    assert m.test_function("abcd") == "test_function(char *)"
    assert m.test_function(1, 1.0) == "test_function(int, float)"
    assert m.test_function(1, 1.0) == "test_function(int, float)"
    assert m.test_function(2.0, 2) == "test_function(float, int)"


def test_bytes():
    assert m.print_bytes(m.return_bytes()) == "bytes[1 0 2 0]"


def test_exception_specifiers():
    c = m.C()
    assert c.m1(2) == 1
    assert c.m2(3) == 1
    assert c.m3(5) == 2
    assert c.m4(7) == 3
    assert c.m5(10) == 5
    assert c.m6(14) == 8
    assert c.m7(20) == 13
    assert c.m8(29) == 21

    assert m.f1(33) == 34
    assert m.f2(53) == 55
    assert m.f3(86) == 89
    assert m.f4(140) == 144


def test_function_record_leaks():
    class RaisingRepr:
        def __repr__(self):
            raise RuntimeError("Surprise!")

    with pytest.raises(RuntimeError):
        m.register_large_capture_with_invalid_arguments(m)
    with pytest.raises(RuntimeError):
        m.register_with_raising_repr(m, RaisingRepr())
