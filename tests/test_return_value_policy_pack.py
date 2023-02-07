import pytest

from pybind11_tests import return_value_policy_pack as m


@pytest.mark.parametrize(
    "func, expected",
    [
        (m.return_tuple_str_str, (str, str)),
        (m.return_tuple_bytes_bytes, (bytes, bytes)),
        (m.return_tuple_str_bytes, (str, bytes)),
        (m.return_tuple_bytes_str, (bytes, str)),
    ],
)
def test_return_pair_string(func, expected):
    p = func()
    assert isinstance(p, tuple)
    assert len(p) == 2
    assert all(isinstance(e, t) for e, t in zip(p, expected))


@pytest.mark.parametrize(
    "func, expected",
    [
        (m.return_nested_tuple_str, (str, str, str, str)),
        (m.return_nested_tuple_bytes, (bytes, bytes, bytes, bytes)),
        (m.return_nested_tuple_str_bytes_bytes_str, (str, bytes, bytes, str)),
        (m.return_nested_tuple_bytes_str_str_bytes, (bytes, str, str, bytes)),
    ],
)
def test_return_nested_pair_string(func, expected):
    np = func()
    assert isinstance(np, tuple)
    assert len(np) == 2
    assert all(isinstance(e, t) for e, t in zip(sum(np, ()), expected))


@pytest.mark.parametrize(
    "func, expected",
    [
        (m.return_dict_str_str, (str, str)),
        (m.return_dict_bytes_bytes, (bytes, bytes)),
        (m.return_dict_str_bytes, (str, bytes)),
        (m.return_dict_bytes_str, (bytes, str)),
    ],
)
def test_return_map_string(func, expected):
    mp = func()
    assert isinstance(mp, dict)
    assert len(mp) == 1
    assert all(isinstance(e, t) for e, t in zip(tuple(mp.items())[0], expected))


@pytest.mark.parametrize(
    "func, expected",
    [
        (m.return_dict_sbbs, (str, bytes, bytes, str)),
        (m.return_dict_bssb, (bytes, str, str, bytes)),
    ],
)
def test_return_map_pair_string(func, expected):
    mp = func()
    assert isinstance(mp, dict)
    assert len(mp) == 1
    assert all(
        isinstance(e, t) for e, t in zip(sum(tuple(mp.items())[0], ()), expected)
    )
