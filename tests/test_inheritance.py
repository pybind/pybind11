

def test_automatic_upcasting():
    from pybind11_tests import return_class_1, return_class_2, return_none

    assert type(return_class_1()).__name__ == "DerivedClass1"
    assert type(return_class_2()).__name__ == "DerivedClass2"
    assert type(return_none()).__name__ == "NoneType"
