

def test_constants():
    from pybind11_tests import some_constant

    assert some_constant == 14


def test_function_overloading():
    from pybind11_tests import MyEnum, test_function

    assert test_function() == "test_function()"
    assert test_function(7) == "test_function(7)"
    assert test_function(MyEnum.EFirstEntry) == "test_function(enum=1)"
    assert test_function(MyEnum.ESecondEntry) == "test_function(enum=2)"


def test_bytes():
    from pybind11_tests import return_bytes, print_bytes

    assert print_bytes(return_bytes()) == "bytes[1 0 2 0]"
