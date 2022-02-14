import class_sh_module_local_0 as m0
import class_sh_module_local_1 as m1
import class_sh_module_local_2 as m2
import pytest


def test_cross_module_get_mtxt():
    obj1 = m1.atyp("A")
    assert obj1.tag() == 1
    obj2 = m2.atyp("B")
    assert obj2.tag() == 2
    assert m1.get_mtxt(obj1) == "A"
    assert m2.get_mtxt(obj2) == "B"
    assert m1.get_mtxt(obj2) == "B"
    assert m2.get_mtxt(obj1) == "A"
    assert m0.get_mtxt(obj1) == "A"
    assert m0.get_mtxt(obj2) == "B"


def test_m0_rtrn_valu_atyp():
    with pytest.raises(TypeError) as exc_info:
        m0.rtrn_valu_atyp()
    assert str(exc_info.value).startswith(
        "Unable to convert function return value to a Python type!"
    )
