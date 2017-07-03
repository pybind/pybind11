# Python < 3 needs this: coding=utf-8
import pytest

from pybind11_tests import builtin_casters as m
from pybind11_tests import UserType, IncType


def test_simple_string():
    assert m.string_roundtrip("const char *") == "const char *"


def test_unicode_conversion():
    """Tests unicode conversion and error reporting."""
    assert m.good_utf8_string() == u"Say utf8‽ 🎂 𝐀"
    assert m.good_utf16_string() == u"b‽🎂𝐀z"
    assert m.good_utf32_string() == u"a𝐀🎂‽z"
    assert m.good_wchar_string() == u"a⸘𝐀z"

    with pytest.raises(UnicodeDecodeError):
        m.bad_utf8_string()

    with pytest.raises(UnicodeDecodeError):
        m.bad_utf16_string()

    # These are provided only if they actually fail (they don't when 32-bit and under Python 2.7)
    if hasattr(m, "bad_utf32_string"):
        with pytest.raises(UnicodeDecodeError):
            m.bad_utf32_string()
    if hasattr(m, "bad_wchar_string"):
        with pytest.raises(UnicodeDecodeError):
            m.bad_wchar_string()

    assert m.u8_Z() == 'Z'
    assert m.u8_eacute() == u'é'
    assert m.u16_ibang() == u'‽'
    assert m.u32_mathbfA() == u'𝐀'
    assert m.wchar_heart() == u'♥'


def test_single_char_arguments():
    """Tests failures for passing invalid inputs to char-accepting functions"""
    def toobig_message(r):
        return "Character code point not in range({0:#x})".format(r)
    toolong_message = "Expected a character, but multi-character string found"

    assert m.ord_char(u'a') == 0x61  # simple ASCII
    assert m.ord_char(u'é') == 0xE9  # requires 2 bytes in utf-8, but can be stuffed in a char
    with pytest.raises(ValueError) as excinfo:
        assert m.ord_char(u'Ā') == 0x100  # requires 2 bytes, doesn't fit in a char
    assert str(excinfo.value) == toobig_message(0x100)
    with pytest.raises(ValueError) as excinfo:
        assert m.ord_char(u'ab')
    assert str(excinfo.value) == toolong_message

    assert m.ord_char16(u'a') == 0x61
    assert m.ord_char16(u'é') == 0xE9
    assert m.ord_char16(u'Ā') == 0x100
    assert m.ord_char16(u'‽') == 0x203d
    assert m.ord_char16(u'♥') == 0x2665
    with pytest.raises(ValueError) as excinfo:
        assert m.ord_char16(u'🎂') == 0x1F382  # requires surrogate pair
    assert str(excinfo.value) == toobig_message(0x10000)
    with pytest.raises(ValueError) as excinfo:
        assert m.ord_char16(u'aa')
    assert str(excinfo.value) == toolong_message

    assert m.ord_char32(u'a') == 0x61
    assert m.ord_char32(u'é') == 0xE9
    assert m.ord_char32(u'Ā') == 0x100
    assert m.ord_char32(u'‽') == 0x203d
    assert m.ord_char32(u'♥') == 0x2665
    assert m.ord_char32(u'🎂') == 0x1F382
    with pytest.raises(ValueError) as excinfo:
        assert m.ord_char32(u'aa')
    assert str(excinfo.value) == toolong_message

    assert m.ord_wchar(u'a') == 0x61
    assert m.ord_wchar(u'é') == 0xE9
    assert m.ord_wchar(u'Ā') == 0x100
    assert m.ord_wchar(u'‽') == 0x203d
    assert m.ord_wchar(u'♥') == 0x2665
    if m.wchar_size == 2:
        with pytest.raises(ValueError) as excinfo:
            assert m.ord_wchar(u'🎂') == 0x1F382  # requires surrogate pair
        assert str(excinfo.value) == toobig_message(0x10000)
    else:
        assert m.ord_wchar(u'🎂') == 0x1F382
    with pytest.raises(ValueError) as excinfo:
        assert m.ord_wchar(u'aa')
    assert str(excinfo.value) == toolong_message


def test_bytes_to_string():
    """Tests the ability to pass bytes to C++ string-accepting functions.  Note that this is
    one-way: the only way to return bytes to Python is via the pybind11::bytes class."""
    # Issue #816
    import sys
    byte = bytes if sys.version_info[0] < 3 else str

    assert m.strlen(byte("hi")) == 2
    assert m.string_length(byte("world")) == 5
    assert m.string_length(byte("a\x00b")) == 3
    assert m.strlen(byte("a\x00b")) == 1  # C-string limitation

    # passing in a utf8 encoded string should work
    assert m.string_length(u'💩'.encode("utf8")) == 4


@pytest.mark.skipif(not hasattr(m, "has_string_view"), reason="no <string_view>")
def test_string_view(capture):
    """Tests support for C++17 string_view arguments and return values"""
    assert m.string_view_chars("Hi") == [72, 105]
    assert m.string_view_chars("Hi 🎂") == [72, 105, 32, 0xf0, 0x9f, 0x8e, 0x82]
    assert m.string_view16_chars("Hi 🎂") == [72, 105, 32, 0xd83c, 0xdf82]
    assert m.string_view32_chars("Hi 🎂") == [72, 105, 32, 127874]

    assert m.string_view_return() == "utf8 secret 🎂"
    assert m.string_view16_return() == "utf16 secret 🎂"
    assert m.string_view32_return() == "utf32 secret 🎂"

    with capture:
        m.string_view_print("Hi")
        m.string_view_print("utf8 🎂")
        m.string_view16_print("utf16 🎂")
        m.string_view32_print("utf32 🎂")
    assert capture == """
        Hi 2
        utf8 🎂 9
        utf16 🎂 8
        utf32 🎂 7
    """

    with capture:
        m.string_view_print("Hi, ascii")
        m.string_view_print("Hi, utf8 🎂")
        m.string_view16_print("Hi, utf16 🎂")
        m.string_view32_print("Hi, utf32 🎂")
    assert capture == """
        Hi, ascii 9
        Hi, utf8 🎂 13
        Hi, utf16 🎂 12
        Hi, utf32 🎂 11
    """


def test_integer_casting():
    """Issue #929 - out-of-range integer values shouldn't be accepted"""
    import sys
    assert m.i32_str(-1) == "-1"
    assert m.i64_str(-1) == "-1"
    assert m.i32_str(2000000000) == "2000000000"
    assert m.u32_str(2000000000) == "2000000000"
    if sys.version_info < (3,):
        assert m.i32_str(long(-1)) == "-1"
        assert m.i64_str(long(-1)) == "-1"
        assert m.i64_str(long(-999999999999)) == "-999999999999"
        assert m.u64_str(long(999999999999)) == "999999999999"
    else:
        assert m.i64_str(-999999999999) == "-999999999999"
        assert m.u64_str(999999999999) == "999999999999"

    with pytest.raises(TypeError) as excinfo:
        m.u32_str(-1)
    assert "incompatible function arguments" in str(excinfo.value)
    with pytest.raises(TypeError) as excinfo:
        m.u64_str(-1)
    assert "incompatible function arguments" in str(excinfo.value)
    with pytest.raises(TypeError) as excinfo:
        m.i32_str(-3000000000)
    assert "incompatible function arguments" in str(excinfo.value)
    with pytest.raises(TypeError) as excinfo:
        m.i32_str(3000000000)
    assert "incompatible function arguments" in str(excinfo.value)

    if sys.version_info < (3,):
        with pytest.raises(TypeError) as excinfo:
            m.u32_str(long(-1))
        assert "incompatible function arguments" in str(excinfo.value)
        with pytest.raises(TypeError) as excinfo:
            m.u64_str(long(-1))
        assert "incompatible function arguments" in str(excinfo.value)


def test_tuple(doc):
    """std::pair <-> tuple & std::tuple <-> tuple"""
    assert m.pair_passthrough((True, "test")) == ("test", True)
    assert m.tuple_passthrough((True, "test", 5)) == (5, "test", True)
    # Any sequence can be cast to a std::pair or std::tuple
    assert m.pair_passthrough([True, "test"]) == ("test", True)
    assert m.tuple_passthrough([True, "test", 5]) == (5, "test", True)
    assert m.empty_tuple() == ()

    assert doc(m.pair_passthrough) == """
        pair_passthrough(arg0: Tuple[bool, str]) -> Tuple[str, bool]

        Return a pair in reversed order
    """
    assert doc(m.tuple_passthrough) == """
        tuple_passthrough(arg0: Tuple[bool, str, int]) -> Tuple[int, str, bool]

        Return a triple in reversed order
    """

    assert m.rvalue_pair() == ("rvalue", "rvalue")
    assert m.lvalue_pair() == ("lvalue", "lvalue")
    assert m.rvalue_tuple() == ("rvalue", "rvalue", "rvalue")
    assert m.lvalue_tuple() == ("lvalue", "lvalue", "lvalue")
    assert m.rvalue_nested() == ("rvalue", ("rvalue", ("rvalue", "rvalue")))
    assert m.lvalue_nested() == ("lvalue", ("lvalue", ("lvalue", "lvalue")))


def test_builtins_cast_return_none():
    """Casters produced with PYBIND11_TYPE_CASTER() should convert nullptr to None"""
    assert m.return_none_string() is None
    assert m.return_none_char() is None
    assert m.return_none_bool() is None
    assert m.return_none_int() is None
    assert m.return_none_float() is None


def test_none_deferred():
    """None passed as various argument types should defer to other overloads"""
    assert not m.defer_none_cstring("abc")
    assert m.defer_none_cstring(None)
    assert not m.defer_none_custom(UserType())
    assert m.defer_none_custom(None)
    assert m.nodefer_none_void(None)


def test_void_caster():
    assert m.load_nullptr_t(None) is None
    assert m.cast_nullptr_t() is None


def test_reference_wrapper():
    """std::reference_wrapper for builtin and user types"""
    assert m.refwrap_builtin(42) == 420
    assert m.refwrap_usertype(UserType(42)) == 42

    with pytest.raises(TypeError) as excinfo:
        m.refwrap_builtin(None)
    assert "incompatible function arguments" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        m.refwrap_usertype(None)
    assert "incompatible function arguments" in str(excinfo.value)

    a1 = m.refwrap_list(copy=True)
    a2 = m.refwrap_list(copy=True)
    assert [x.value for x in a1] == [2, 3]
    assert [x.value for x in a2] == [2, 3]
    assert not a1[0] is a2[0] and not a1[1] is a2[1]

    b1 = m.refwrap_list(copy=False)
    b2 = m.refwrap_list(copy=False)
    assert [x.value for x in b1] == [1, 2]
    assert [x.value for x in b2] == [1, 2]
    assert b1[0] is b2[0] and b1[1] is b2[1]

    assert m.refwrap_iiw(IncType(5)) == 5
    assert m.refwrap_call_iiw(IncType(10), m.refwrap_iiw) == [10, 10, 10, 10]


def test_complex_cast():
    """std::complex casts"""
    assert m.complex_cast(1) == "1.0"
    assert m.complex_cast(2j) == "(0.0, 2.0)"
