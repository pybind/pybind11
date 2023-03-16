import pytest

from pybind11_tests import return_value_policy_pack as m


@pytest.mark.parametrize(
    ("func", "expected"),
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
    ("func", "expected"),
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
    ("func", "expected"),
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
    ("func", "expected"),
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
    ("func", "expected"),
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
    ("func", "expected"),
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
    ("fname", "expected"),
    [
        ("return_optional_sb", (str, bytes)),
        ("return_optional_bs", (bytes, str)),
    ],
)
def test_return_optional_pair_string(fname, expected):
    _test_return_optional_variant_pair_string(fname, expected)


@pytest.mark.skipif(not m.PYBIND11_HAS_VARIANT, reason="<variant> not available.")
@pytest.mark.parametrize(
    ("fname", "expected"),
    [
        ("return_variant_sb", (str, bytes)),
        ("return_variant_bs", (bytes, str)),
    ],
)
def test_return_variant_pair_string(fname, expected):
    _test_return_optional_variant_pair_string(fname, expected)


@pytest.mark.parametrize(
    ("func", "expected"),
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


def test_nested_callbacks_si_s():
    def cb_1(i):
        assert isinstance(i, int)
        return "cb_1_" + str(i)

    def cb_2(cb):
        r = cb(20)
        assert isinstance(r, str)
        return "cb_2_" + r

    def cb_3(cb):
        r = cb(cb_1)
        assert isinstance(r, str)
        return "cb_3_" + r

    def cb_4(cb):
        r = cb(cb_2)
        assert isinstance(r, str)
        return "cb_4_" + r

    assert m.call_level_1_callback_si_s(cb_1) == "cb_1_1001"
    assert m.call_level_2_callback_si_s(cb_2) == "cb_2_level_0_si_20"
    assert m.call_level_3_callback_si_s(cb_3) == "cb_3_cb_1_1001"
    assert m.call_level_4_callback_si_s(cb_4) == "cb_4_cb_2_level_0_si_20"


def test_nested_callbacks_si_b():
    def cb_1(i):
        assert isinstance(i, int)
        return b"\x80cb_1_" + (str(i)).encode()

    def cb_2(cb):
        r = cb(20)
        assert isinstance(r, bytes)
        return b"\x80cb_2_" + r

    def cb_3(cb):
        r = cb(cb_1)
        assert isinstance(r, bytes)
        return b"\x80cb_3_" + r

    def cb_2s(cb):
        r = cb(20)
        assert isinstance(r, str)  # Passing bytes is unsupported (missing feature).
        return b"\x80cb_2_" + r.encode()

    def cb_4(cb):
        r = cb(cb_2s)
        assert isinstance(r, bytes)
        return b"\x80cb_4_" + r

    assert m.call_level_1_callback_si_b(cb_1) == b"\x80cb_1_1001"
    assert m.call_level_2_callback_si_b(cb_2) == b"\x80cb_2_level_0_si_20"
    assert m.call_level_3_callback_si_b(cb_3) == b"\x80cb_3_\x80cb_1_1001"
    assert m.call_level_4_callback_si_b(cb_4) == b"\x80cb_4_\x80cb_2_level_0_si_20"


def test_nested_callbacks_is_s():
    def cb_1(s):
        assert isinstance(s, str)
        return 10000 + int(s)

    def cb_2(cb):
        return 20000 + cb("20")

    def cb_3(cb):
        return 30000 + cb(cb_1)

    def cb_4(cb):
        return 40000 + cb(cb_2)

    assert m.call_level_1_callback_is_s(cb_1) == 10101
    assert m.call_level_2_callback_is_s(cb_2) == 20120
    assert m.call_level_3_callback_is_s(cb_3) == 40101
    assert m.call_level_4_callback_is_s(cb_4) == 60120


def test_nested_callbacks_is_b():
    def cb_1(s):
        assert isinstance(s, bytes)
        return 10000 + int(s)

    def cb_2(cb):
        return 20000 + cb(b"20")

    def cb_1s(s):
        assert isinstance(s, str)  # Passing bytes is unsupported (missing feature).
        return 10000 + int(s)

    def cb_3(cb):
        return 30000 + cb(cb_1s)

    def cb_4(cb):
        return 40000 + cb(cb_2)

    assert m.call_level_1_callback_is_b(cb_1) == 10101
    assert m.call_level_2_callback_is_b(cb_2) == 20120
    assert m.call_level_3_callback_is_b(cb_3) == 40101
    assert m.call_level_4_callback_is_b(cb_4) == 60120
