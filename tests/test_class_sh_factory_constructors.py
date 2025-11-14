from __future__ import annotations

import pytest

from pybind11_tests import class_sh_factory_constructors as m


def test_atyp_factories():
    assert m.atyp_valu().get_mtxt() == "Valu"
    assert m.atyp_rref().get_mtxt() == "Rref"
    # sert m.atyp_cref().get_mtxt() == "Cref"
    # sert m.atyp_mref().get_mtxt() == "Mref"
    # sert m.atyp_cptr().get_mtxt() == "Cptr"
    assert m.atyp_mptr().get_mtxt() == "Mptr"
    assert m.atyp_shmp().get_mtxt() == "Shmp"
    assert m.atyp_shcp().get_mtxt() == "Shcp"
    assert m.atyp_uqmp().get_mtxt() == "Uqmp"
    assert m.atyp_uqcp().get_mtxt() == "Uqcp"
    assert m.atyp_udmp().get_mtxt() == "Udmp"
    assert m.atyp_udcp().get_mtxt() == "Udcp"


@pytest.mark.parametrize(
    ("init_args", "expected"),
    [
        ((3,), 300),
        ((5, 7), 570),
        ((9, 11, 13), 1023),
    ],
)
def test_with_alias_success(init_args, expected):
    assert m.with_alias(*init_args).val == expected


@pytest.mark.parametrize(
    ("num_init_args", "smart_ptr"),
    [
        (4, "std::unique_ptr"),
        (5, "std::shared_ptr"),
    ],
)
def test_with_alias_invalid(num_init_args, smart_ptr):
    class PyDrvdWithAlias(m.with_alias):
        pass

    with pytest.raises(TypeError) as excinfo:
        PyDrvdWithAlias(*((0,) * num_init_args))
    assert (
        str(excinfo.value)
        == "pybind11::init(): construction failed: returned "
        + smart_ptr
        + " pointee is not an alias instance"
    )
