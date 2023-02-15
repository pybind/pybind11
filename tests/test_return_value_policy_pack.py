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
        (m.return_map_sbbs, (str, bytes, bytes, str)),
        (m.return_map_bssb, (bytes, str, str, bytes)),
    ],
)
def test_return_dict_pair_string(func, expected):
    mp = func()
    assert isinstance(mp, dict)
    assert len(mp) == 1
    assert all(
        isinstance(e, t) for e, t in zip(sum(tuple(mp.items())[0], ()), expected)
    )


@pytest.mark.parametrize(
    "func, expected",
    [
        (m.return_set_sb, (str, bytes)),
        (m.return_set_bs, (bytes, str)),
    ],
)
def test_return_set_pair_string(func, expected):
    sp = func()
    assert isinstance(sp, set)
    assert len(sp) == 1
    assert all(isinstance(e, t) for e, t in zip(sum(tuple(sp), ()), expected))


@pytest.mark.parametrize(
    "func, expected",
    [
        (m.return_vector_sb, (str, bytes)),
        (m.return_vector_bs, (bytes, str)),
        (m.return_array_sb, (str, bytes)),
        (m.return_array_bs, (bytes, str)),
    ],
)
def test_return_list_pair_string(func, expected):
    vp = func()
    assert isinstance(vp, list)
    assert len(vp) == 1
    assert all(isinstance(e, t) for e, t in zip(sum(vp, ()), expected))


def _test_return_optional_variant_pair_string(fname, expected):
    func = getattr(m, fname)
    p = func()
    assert isinstance(p, tuple)
    assert len(p) == 2
    assert all(isinstance(e, t) for e, t in zip(p, expected))


@pytest.mark.skipif(not m.PYBIND11_HAS_OPTIONAL, reason="<optional> not available.")
@pytest.mark.parametrize(
    "fname, expected",
    [
        ("return_optional_sb", (str, bytes)),
        ("return_optional_bs", (bytes, str)),
    ],
)
def test_return_optional_pair_string(fname, expected):
    _test_return_optional_variant_pair_string(fname, expected)


@pytest.mark.skipif(not m.PYBIND11_HAS_VARIANT, reason="<variant> not available.")
@pytest.mark.parametrize(
    "fname, expected",
    [
        ("return_variant_sb", (str, bytes)),
        ("return_variant_bs", (bytes, str)),
    ],
)
def test_return_variant_pair_string(fname, expected):
    _test_return_optional_variant_pair_string(fname, expected)


@pytest.mark.parametrize(
    "func, expected",
    [
        (m.call_callback_pass_pair_default, repr(("", ""))),
        (m.call_callback_pass_pair_s, repr(("", ""))),
        (m.call_callback_pass_pair_b, repr((b"", b""))),
        (m.call_callback_pass_pair_sb, repr(("", b""))),
        (m.call_callback_pass_pair_bs, repr((b"", ""))),
    ],
)
def test_call_callback_pass_pair_string(func, expected):
    def cb(p):
        assert isinstance(p, tuple)
        assert len(p) == 2
        return repr(p)

    assert func(cb) == expected


@pytest.mark.parametrize(
    "func, inner_arg, expected",
    [
        (m.nested_callbacks_rtn_s, 23, "-23"),
        (m.nested_callbacks_rtn_b, 45, b"-45"),
    ],
)
def test_nested_callbacks_rtn_string(func, inner_arg, expected):
    def cb(inner_cb):
        inner_val = inner_cb(inner_arg)
        assert inner_val == expected
        return inner_val

    val = func(cb)
    assert val == expected
