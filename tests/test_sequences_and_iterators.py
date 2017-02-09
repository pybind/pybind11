import pytest


def isclose(a, b, rel_tol=1e-05, abs_tol=0.0):
    """Like math.isclose() from Python 3.5"""
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def allclose(a_list, b_list, rel_tol=1e-05, abs_tol=0.0):
    return all(isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol) for a, b in zip(a_list, b_list))


def test_generalized_iterators():
    from pybind11_tests.sequences_and_iterators import IntPairs

    assert list(IntPairs([(1, 2), (3, 4), (0, 5)]).nonzero()) == [(1, 2), (3, 4)]
    assert list(IntPairs([(1, 2), (2, 0), (0, 3), (4, 5)]).nonzero()) == [(1, 2)]
    assert list(IntPairs([(0, 3), (1, 2), (3, 4)]).nonzero()) == []

    assert list(IntPairs([(1, 2), (3, 4), (0, 5)]).nonzero_keys()) == [1, 3]
    assert list(IntPairs([(1, 2), (2, 0), (0, 3), (4, 5)]).nonzero_keys()) == [1]
    assert list(IntPairs([(0, 3), (1, 2), (3, 4)]).nonzero_keys()) == []


def test_sequence():
    from pybind11_tests import ConstructorStats
    from pybind11_tests.sequences_and_iterators import Sequence

    cstats = ConstructorStats.get(Sequence)

    s = Sequence(5)
    assert cstats.values() == ['of size', '5']

    assert "Sequence" in repr(s)
    assert len(s) == 5
    assert s[0] == 0 and s[3] == 0
    assert 12.34 not in s
    s[0], s[3] = 12.34, 56.78
    assert 12.34 in s
    assert isclose(s[0], 12.34) and isclose(s[3], 56.78)

    rev = reversed(s)
    assert cstats.values() == ['of size', '5']

    rev2 = s[::-1]
    assert cstats.values() == ['of size', '5']

    expected = [0, 56.78, 0, 0, 12.34]
    assert allclose(rev, expected)
    assert allclose(rev2, expected)
    assert rev == rev2

    rev[0::2] = Sequence([2.0, 2.0, 2.0])
    assert cstats.values() == ['of size', '3', 'from std::vector']

    assert allclose(rev, [2, 56.78, 2, 0, 2])

    assert cstats.alive() == 3
    del s
    assert cstats.alive() == 2
    del rev
    assert cstats.alive() == 1
    del rev2
    assert cstats.alive() == 0

    assert cstats.values() == []
    assert cstats.default_constructions == 0
    assert cstats.copy_constructions == 0
    assert cstats.move_constructions >= 1
    assert cstats.copy_assignments == 0
    assert cstats.move_assignments == 0


def test_map_iterator():
    from pybind11_tests.sequences_and_iterators import StringMap

    m = StringMap({'hi': 'bye', 'black': 'white'})
    assert m['hi'] == 'bye'
    assert len(m) == 2
    assert m['black'] == 'white'

    with pytest.raises(KeyError):
        assert m['orange']
    m['orange'] = 'banana'
    assert m['orange'] == 'banana'

    expected = {'hi': 'bye', 'black': 'white', 'orange': 'banana'}
    for k in m:
        assert m[k] == expected[k]
    for k, v in m.items():
        assert v == expected[k]


def test_python_iterator_in_cpp():
    import pybind11_tests.sequences_and_iterators as m

    t = (1, 2, 3)
    assert m.object_to_list(t) == [1, 2, 3]
    assert m.object_to_list(iter(t)) == [1, 2, 3]
    assert m.iterator_to_list(iter(t)) == [1, 2, 3]

    with pytest.raises(TypeError) as excinfo:
        m.object_to_list(1)
    assert "object is not iterable" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        m.iterator_to_list(1)
    assert "incompatible function arguments" in str(excinfo.value)

    def bad_next_call():
        raise RuntimeError("py::iterator::advance() should propagate errors")

    with pytest.raises(RuntimeError) as excinfo:
        m.iterator_to_list(iter(bad_next_call, None))
    assert str(excinfo.value) == "py::iterator::advance() should propagate errors"

    l = [1, None, 0, None]
    assert m.count_none(l) == 2
    assert m.find_none(l) is True
    assert m.count_nonzeros({"a": 0, "b": 1, "c": 2}) == 2

    r = range(5)
    assert all(m.tuple_iterator(tuple(r)))
    assert all(m.list_iterator(list(r)))
    assert all(m.sequence_iterator(r))
