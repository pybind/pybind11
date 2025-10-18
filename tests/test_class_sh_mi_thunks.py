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


def test_virtual_base_not_at_offset_0():
    # This test ensures that the Diamond fixture actually exercises a non-zero
    # virtual-base subobject offset on our supported platforms/ABIs.
    #
    # If this assert ever fails on some platform/toolchain, please adjust the
    # C++ fixture so the virtual base is *not* at offset 0:
    #   - Keep VBase non-empty.
    #   - Make Left and Right non-empty and asymmetrically sized and, if
    #     needed, nudge with a modest alignment.
    #   - The goal is to achieve a non-zero address delta between `Diamond*`
    #     and `static_cast<VBase*>(Diamond*)`.
    #
    # Rationale: certain smart_holder features are exercised only when the
    # registered subobject address differs from the most-derived object start,
    # so this check guards test efficacy across compilers.
    addrs = m.diamond_addrs()
    assert addrs.as_vbase - addrs.as_self != 0, (
        "Diamond VBase at offset 0 on this platform; to ensure test efficacy, "
        "tweak fixtures (VBase/Left/Right) to ensure non-zero subobject offset."
    )


@pytest.mark.parametrize(
    "make_fn",
    [
        m.make_diamond_as_vbase_raw_ptr,  # exercises smart_holder::from_raw_ptr_take_ownership
        m.make_diamond_as_vbase_shared_ptr,  # exercises smart_holder_from_shared_ptr
        m.make_diamond_as_vbase_unique_ptr,  # exercises smart_holder_from_unique_ptr
    ],
)
def test_make_diamond_as_vbase(make_fn):
    # Added under PR #5836
    vb = make_fn()
    assert vb.ping() == 7


@pytest.mark.parametrize(
    "clone_fn",
    [
        m.Tiger.clone_raw_ptr,
        m.Tiger.clone_shared_ptr,
        m.Tiger.clone_unique_ptr,
    ],
)
def test_animal_cat_tiger(clone_fn):
    # Based on Animal-Cat-Tiger reproducer under PR #5796
    tiger = m.Tiger()
    cloned = clone_fn(tiger)
    assert isinstance(cloned, m.Tiger)
