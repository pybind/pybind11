

def test_constants():
    from pybind11_tests import some_constant

    assert some_constant == 14


def test_function_overloading():
    from pybind11_tests import MyEnum, test_function

    assert test_function() == "test_function()"
    assert test_function(7) == "test_function(7)"
    assert test_function(MyEnum.EFirstEntry) == "test_function(enum=1)"
    assert test_function(MyEnum.ESecondEntry) == "test_function(enum=2)"

    assert test_function(1, 1.0) == "test_function(int, float)"
    assert test_function(2.0, 2) == "test_function(float, int)"


def test_bytes():
    from pybind11_tests import return_bytes, print_bytes

    assert print_bytes(return_bytes()) == "bytes[1 0 2 0]"


def test_exception_specifiers():
    from pybind11_tests.exc_sp import C, f1, f2, f3, f4

    c = C()
    assert c.m1(2) == 1
    assert c.m2(3) == 1
    assert c.m3(5) == 2
    assert c.m4(7) == 3
    assert c.m5(10) == 5
    assert c.m6(14) == 8
    assert c.m7(20) == 13
    assert c.m8(29) == 21

    assert f1(33) == 34
    assert f2(53) == 55
    assert f3(86) == 89
    assert f4(140) == 144
