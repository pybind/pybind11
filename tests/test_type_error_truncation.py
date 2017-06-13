import pytest
import pybind11_tests






def test_type_error_truncation():
    from pybind11_tests import TypeWithLongRepr

    size_of_repr = 200
    objA = TypeWithLongRepr(200)
    objB = TypeWithLongRepr(50)
    objC = TypeWithLongRepr(150)
    reprStrA = repr(objA)
    reprStrB = repr(objB)
    reprStrC = repr(objC)
    #print(reprStr)
    assert len(reprStrA) == 200
    assert len(reprStrB) == 50
    assert len(reprStrC) == 150


    assert reprStrA.startswith('<')
    assert reprStrA.endswith('>')



    with pytest.raises(TypeError) as excinfo:
        objA.foo(objC, objC)
    typeErrorMsg = str(excinfo.value)
    print(typeErrorMsg)
    assert len(typeErrorMsg) < 3 * 200
    assert typeErrorMsg.contains('...[truncated by pybind11]')


    with pytest.raises(TypeError) as excinfo:
        objB.foo(objB, objB)
    typeErrorMsg = str(excinfo.value)
    print(typeErrorMsg)
    assert len(typeErrorMsg) < 3 * 200
    assert not typeErrorMsg.contains('...[truncated by pybind11]')





    with pytest.raises(TypeError) as excinfo:
        objA.foo(objC, arg2=objC)
    typeErrorMsg = str(excinfo.value)
    print(typeErrorMsg)
    assert len(typeErrorMsg) < 3 * 200
    assert typeErrorMsg.contains('...[truncated by pybind11]')


    with pytest.raises(TypeError) as excinfo:
        objB.foo(objB, objB)
    typeErrorMsg = str(excinfo.value)
    print(typeErrorMsg)
    assert len(typeErrorMsg) < 3 * 200
    assert not typeErrorMsg.contains('...[truncated by pybind11]')




test_type_error_truncation()