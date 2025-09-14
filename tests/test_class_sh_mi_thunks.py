from __future__ import annotations

import pytest

from pybind11_tests import class_sh_mi_thunks as m


def test_ptrdiff_drvd_base0():
    ptrdiff = m.ptrdiff_drvd_base0()
    # A failure here does not (necessarily) mean that there is a bug, but that
    # test_class_sh_mi_thunks is not exercising what it is supposed to.
    # If this ever fails on some platforms: use pytest.skip()
    # If this ever fails on all platforms: don't know, seems extremely unlikely.
    assert ptrdiff != 0


@pytest.mark.parametrize(
    "vec_size_fn",
    [
        m.vec_size_base0_raw_ptr,
        m.vec_size_base0_shared_ptr,
    ],
)
@pytest.mark.parametrize(
    "get_fn",
    [
        m.get_drvd_as_base0_raw_ptr,
        m.get_drvd_as_base0_shared_ptr,
        m.get_drvd_as_base0_unique_ptr,
    ],
)
def test_get_vec_size_raw_shared(get_fn, vec_size_fn):
    obj = get_fn()
    assert vec_size_fn(obj) == 5


@pytest.mark.parametrize(
    "get_fn", [m.get_drvd_as_base0_raw_ptr, m.get_drvd_as_base0_unique_ptr]
)
def test_get_vec_size_unique(get_fn):
    obj = get_fn()
    assert m.vec_size_base0_unique_ptr(obj) == 5
    with pytest.raises(ValueError, match="Python instance was disowned"):
        m.vec_size_base0_unique_ptr(obj)


def test_get_shared_vec_size_unique():
    obj = m.get_drvd_as_base0_shared_ptr()
    with pytest.raises(ValueError) as exc_info:
        m.vec_size_base0_unique_ptr(obj)
    assert (
        str(exc_info.value) == "Cannot disown external shared_ptr (load_as_unique_ptr)."
    )


def test_virtual_base_at_offset_0():
    addrs = m.diamond_addrs()
    if addrs.as_vbase - addrs.as_self == 0:
        pytest.skip("virtual base at offset 0 on this compiler/layout")


def test_shared_ptr_return_to_virtual_base_triggers_vi_path():
    # This exercised the broken seam pre-fix.
    vb = m.make_diamond_as_vbase()
    # If registration used the wrong subobject, this would crash pre-fix on MSVC
    # and often on Linux with diamond MI (or under ASan/UBSan).
    assert vb.ping() == 7
