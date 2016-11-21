import re
import pytest

pytestmark = pytest.requires_numpy

with pytest.suppress(ImportError):
    import numpy as np


@pytest.fixture(scope='module')
def simple_dtype():
    ld = np.dtype('longdouble')
    return np.dtype({'names': ['bool_', 'uint_', 'float_', 'ldbl_'],
                     'formats': ['?', 'u4', 'f4', 'f{}'.format(ld.itemsize)],
                     'offsets': [0, 4, 8, (16 if ld.alignment > 4 else 12)]})


@pytest.fixture(scope='module')
def packed_dtype():
    return np.dtype([('bool_', '?'), ('uint_', 'u4'), ('float_', 'f4'), ('ldbl_', 'g')])


def dt_fmt():
    from sys import byteorder
    e = '<' if byteorder == 'little' else '>'
    return ("{{'names':['bool_','uint_','float_','ldbl_'],"
            " 'formats':['?','" + e + "u4','" + e + "f4','" + e + "f{}'],"
            " 'offsets':[0,4,8,{}], 'itemsize':{}}}")


def simple_dtype_fmt():
    ld = np.dtype('longdouble')
    simple_ld_off = 12 + 4 * (ld.alignment > 4)
    return dt_fmt().format(ld.itemsize, simple_ld_off, simple_ld_off + ld.itemsize)


def packed_dtype_fmt():
    from sys import byteorder
    return "[('bool_', '?'), ('uint_', '{e}u4'), ('float_', '{e}f4'), ('ldbl_', '{e}f{}')]".format(
        np.dtype('longdouble').itemsize, e='<' if byteorder == 'little' else '>')


def partial_ld_offset():
    return 12 + 4 * (np.dtype('uint64').alignment > 4) + 8 + 8 * (
        np.dtype('longdouble').alignment > 8)


def partial_dtype_fmt():
    ld = np.dtype('longdouble')
    partial_ld_off = partial_ld_offset()
    return dt_fmt().format(ld.itemsize, partial_ld_off, partial_ld_off + ld.itemsize)


def partial_nested_fmt():
    ld = np.dtype('longdouble')
    partial_nested_off = 8 + 8 * (ld.alignment > 8)
    partial_ld_off = partial_ld_offset()
    partial_nested_size = partial_nested_off * 2 + partial_ld_off + ld.itemsize
    return "{{'names':['a'], 'formats':[{}], 'offsets':[{}], 'itemsize':{}}}".format(
        partial_dtype_fmt(), partial_nested_off, partial_nested_size)


def assert_equal(actual, expected_data, expected_dtype):
    np.testing.assert_equal(actual, np.array(expected_data, dtype=expected_dtype))


def test_format_descriptors():
    from pybind11_tests import get_format_unbound, print_format_descriptors

    with pytest.raises(RuntimeError) as excinfo:
        get_format_unbound()
    assert re.match('^NumPy type info missing for .*UnboundStruct.*$', str(excinfo.value))

    ld = np.dtype('longdouble')
    ldbl_fmt = ('4x' if ld.alignment > 4 else '') + ld.char
    ss_fmt = "T{?:bool_:3xI:uint_:f:float_:" + ldbl_fmt + ":ldbl_:}"
    dbl = np.dtype('double')
    partial_fmt = ("T{?:bool_:3xI:uint_:f:float_:" +
                   str(4 * (dbl.alignment > 4) + dbl.itemsize + 8 * (ld.alignment > 8)) +
                   "xg:ldbl_:}")
    nested_extra = str(max(8, ld.alignment))
    assert print_format_descriptors() == [
        ss_fmt,
        "T{?:bool_:^I:uint_:^f:float_:^g:ldbl_:}",
        "T{" + ss_fmt + ":a:T{?:bool_:^I:uint_:^f:float_:^g:ldbl_:}:b:}",
        partial_fmt,
        "T{" + nested_extra + "x" + partial_fmt + ":a:" + nested_extra + "x}",
        "T{3s:a:3s:b:}",
        'T{q:e1:B:e2:}'
    ]


def test_dtype(simple_dtype):
    from pybind11_tests import (print_dtypes, test_dtype_ctors, test_dtype_methods,
                                trailing_padding_dtype, buffer_to_dtype)
    from sys import byteorder
    e = '<' if byteorder == 'little' else '>'

    assert print_dtypes() == [
        simple_dtype_fmt(),
        packed_dtype_fmt(),
        "[('a', {}), ('b', {})]".format(simple_dtype_fmt(), packed_dtype_fmt()),
        partial_dtype_fmt(),
        partial_nested_fmt(),
        "[('a', 'S3'), ('b', 'S3')]",
        "[('e1', '" + e + "i8'), ('e2', 'u1')]",
        "[('x', 'i1'), ('y', '" + e + "u8')]"
    ]

    d1 = np.dtype({'names': ['a', 'b'], 'formats': ['int32', 'float64'],
                   'offsets': [1, 10], 'itemsize': 20})
    d2 = np.dtype([('a', 'i4'), ('b', 'f4')])
    assert test_dtype_ctors() == [np.dtype('int32'), np.dtype('float64'),
                                  np.dtype('bool'), d1, d1, np.dtype('uint32'), d2]

    assert test_dtype_methods() == [np.dtype('int32'), simple_dtype, False, True,
                                    np.dtype('int32').itemsize, simple_dtype.itemsize]

    assert trailing_padding_dtype() == buffer_to_dtype(np.zeros(1, trailing_padding_dtype()))


def test_recarray(simple_dtype, packed_dtype):
    from pybind11_tests import (create_rec_simple, create_rec_packed, create_rec_nested,
                                print_rec_simple, print_rec_packed, print_rec_nested,
                                create_rec_partial, create_rec_partial_nested)

    elements = [(False, 0, 0.0, -0.0), (True, 1, 1.5, -2.5), (False, 2, 3.0, -5.0)]

    for func, dtype in [(create_rec_simple, simple_dtype), (create_rec_packed, packed_dtype)]:
        arr = func(0)
        assert arr.dtype == dtype
        assert_equal(arr, [], simple_dtype)
        assert_equal(arr, [], packed_dtype)

        arr = func(3)
        assert arr.dtype == dtype
        assert_equal(arr, elements, simple_dtype)
        assert_equal(arr, elements, packed_dtype)

        if dtype == simple_dtype:
            assert print_rec_simple(arr) == [
                "s:0,0,0,-0",
                "s:1,1,1.5,-2.5",
                "s:0,2,3,-5"
            ]
        else:
            assert print_rec_packed(arr) == [
                "p:0,0,0,-0",
                "p:1,1,1.5,-2.5",
                "p:0,2,3,-5"
            ]

    nested_dtype = np.dtype([('a', simple_dtype), ('b', packed_dtype)])

    arr = create_rec_nested(0)
    assert arr.dtype == nested_dtype
    assert_equal(arr, [], nested_dtype)

    arr = create_rec_nested(3)
    assert arr.dtype == nested_dtype
    assert_equal(arr, [((False, 0, 0.0, -0.0), (True, 1, 1.5, -2.5)),
                       ((True, 1, 1.5, -2.5), (False, 2, 3.0, -5.0)),
                       ((False, 2, 3.0, -5.0), (True, 3, 4.5, -7.5))], nested_dtype)
    assert print_rec_nested(arr) == [
        "n:a=s:0,0,0,-0;b=p:1,1,1.5,-2.5",
        "n:a=s:1,1,1.5,-2.5;b=p:0,2,3,-5",
        "n:a=s:0,2,3,-5;b=p:1,3,4.5,-7.5"
    ]

    arr = create_rec_partial(3)
    assert str(arr.dtype) == partial_dtype_fmt()
    partial_dtype = arr.dtype
    assert '' not in arr.dtype.fields
    assert partial_dtype.itemsize > simple_dtype.itemsize
    assert_equal(arr, elements, simple_dtype)
    assert_equal(arr, elements, packed_dtype)

    arr = create_rec_partial_nested(3)
    assert str(arr.dtype) == partial_nested_fmt()
    assert '' not in arr.dtype.fields
    assert '' not in arr.dtype.fields['a'][0].fields
    assert arr.dtype.itemsize > partial_dtype.itemsize
    np.testing.assert_equal(arr['a'], create_rec_partial(3))


def test_array_constructors():
    from pybind11_tests import test_array_ctors

    data = np.arange(1, 7, dtype='int32')
    for i in range(8):
        np.testing.assert_array_equal(test_array_ctors(10 + i), data.reshape((3, 2)))
        np.testing.assert_array_equal(test_array_ctors(20 + i), data.reshape((3, 2)))
    for i in range(5):
        np.testing.assert_array_equal(test_array_ctors(30 + i), data)
        np.testing.assert_array_equal(test_array_ctors(40 + i), data)


def test_string_array():
    from pybind11_tests import create_string_array, print_string_array

    arr = create_string_array(True)
    assert str(arr.dtype) == "[('a', 'S3'), ('b', 'S3')]"
    assert print_string_array(arr) == [
        "a='',b=''",
        "a='a',b='a'",
        "a='ab',b='ab'",
        "a='abc',b='abc'"
    ]
    dtype = arr.dtype
    assert arr['a'].tolist() == [b'', b'a', b'ab', b'abc']
    assert arr['b'].tolist() == [b'', b'a', b'ab', b'abc']
    arr = create_string_array(False)
    assert dtype == arr.dtype


def test_enum_array():
    from pybind11_tests import create_enum_array, print_enum_array
    from sys import byteorder
    e = '<' if byteorder == 'little' else '>'

    arr = create_enum_array(3)
    dtype = arr.dtype
    assert dtype == np.dtype([('e1', e + 'i8'), ('e2', 'u1')])
    assert print_enum_array(arr) == [
        "e1=A,e2=X",
        "e1=B,e2=Y",
        "e1=A,e2=X"
    ]
    assert arr['e1'].tolist() == [-1, 1, -1]
    assert arr['e2'].tolist() == [1, 2, 1]
    assert create_enum_array(0).dtype == dtype


def test_signature(doc):
    from pybind11_tests import create_rec_nested

    assert doc(create_rec_nested) == "create_rec_nested(arg0: int) -> numpy.ndarray[NestedStruct]"


def test_scalar_conversion():
    from pybind11_tests import (create_rec_simple, f_simple,
                                create_rec_packed, f_packed,
                                create_rec_nested, f_nested,
                                create_enum_array)

    n = 3
    arrays = [create_rec_simple(n), create_rec_packed(n),
              create_rec_nested(n), create_enum_array(n)]
    funcs = [f_simple, f_packed, f_nested]

    for i, func in enumerate(funcs):
        for j, arr in enumerate(arrays):
            if i == j and i < 2:
                assert [func(arr[k]) for k in range(n)] == [k * 10 for k in range(n)]
            else:
                with pytest.raises(TypeError) as excinfo:
                    func(arr[0])
                assert 'incompatible function arguments' in str(excinfo.value)


def test_register_dtype():
    from pybind11_tests import register_dtype

    with pytest.raises(RuntimeError) as excinfo:
        register_dtype()
    assert 'dtype is already registered' in str(excinfo.value)


@pytest.requires_numpy
def test_compare_buffer_info():
    from pybind11_tests import compare_buffer_info
    assert all(compare_buffer_info())
