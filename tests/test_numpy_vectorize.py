import pytest

with pytest.suppress(ImportError):
    import numpy as np


@pytest.requires_numpy
def test_vectorize(capture):
    from pybind11_tests import vectorized_func, vectorized_func2, vectorized_func3

    assert np.isclose(vectorized_func3(np.array(3 + 7j)), [6 + 14j])

    for f in [vectorized_func, vectorized_func2]:
        with capture:
            assert np.isclose(f(1, 2, 3), 6)
        assert capture == "my_func(x:int=1, y:float=2, z:float=3)"
        with capture:
            assert np.isclose(f(np.array(1), np.array(2), 3), 6)
        assert capture == "my_func(x:int=1, y:float=2, z:float=3)"
        with capture:
            assert np.allclose(f(np.array([1, 3]), np.array([2, 4]), 3), [6, 36])
        assert capture == """
            my_func(x:int=1, y:float=2, z:float=3)
            my_func(x:int=3, y:float=4, z:float=3)
        """
        with capture:
            a, b, c = np.array([[1, 3, 5], [7, 9, 11]]), np.array([[2, 4, 6], [8, 10, 12]]), 3
            assert np.allclose(f(a, b, c), a * b * c)
        assert capture == """
            my_func(x:int=1, y:float=2, z:float=3)
            my_func(x:int=3, y:float=4, z:float=3)
            my_func(x:int=5, y:float=6, z:float=3)
            my_func(x:int=7, y:float=8, z:float=3)
            my_func(x:int=9, y:float=10, z:float=3)
            my_func(x:int=11, y:float=12, z:float=3)
        """
        with capture:
            a, b, c = np.array([[1, 2, 3], [4, 5, 6]]), np.array([2, 3, 4]), 2
            assert np.allclose(f(a, b, c), a * b * c)
        assert capture == """
            my_func(x:int=1, y:float=2, z:float=2)
            my_func(x:int=2, y:float=3, z:float=2)
            my_func(x:int=3, y:float=4, z:float=2)
            my_func(x:int=4, y:float=2, z:float=2)
            my_func(x:int=5, y:float=3, z:float=2)
            my_func(x:int=6, y:float=4, z:float=2)
        """
        with capture:
            a, b, c = np.array([[1, 2, 3], [4, 5, 6]]), np.array([[2], [3]]), 2
            assert np.allclose(f(a, b, c), a * b * c)
        assert capture == """
            my_func(x:int=1, y:float=2, z:float=2)
            my_func(x:int=2, y:float=2, z:float=2)
            my_func(x:int=3, y:float=2, z:float=2)
            my_func(x:int=4, y:float=3, z:float=2)
            my_func(x:int=5, y:float=3, z:float=2)
            my_func(x:int=6, y:float=3, z:float=2)
        """


@pytest.requires_numpy
def test_type_selection():
    from pybind11_tests import selective_func

    assert selective_func(np.array([1], dtype=np.int32)) == "Int branch taken."
    assert selective_func(np.array([1.0], dtype=np.float32)) == "Float branch taken."
    assert selective_func(np.array([1.0j], dtype=np.complex64)) == "Complex float branch taken."


@pytest.requires_numpy
def test_docs(doc):
    from pybind11_tests import vectorized_func

    assert doc(vectorized_func) == """
        vectorized_func(arg0: numpy.ndarray[int], arg1: numpy.ndarray[float], arg2: numpy.ndarray[float]) -> object
    """  # noqa: E501 line too long
