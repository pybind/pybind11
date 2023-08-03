# Importing re before pytest after observing a PyPy CI flake when importing pytest first.
import re

import pytest

from pybind11_tests import class_sh_basic as m


def test_atyp_constructors():
    obj = m.atyp()
    assert obj.__class__.__name__ == "atyp"
    obj = m.atyp("")
    assert obj.__class__.__name__ == "atyp"
    obj = m.atyp("txtm")
    assert obj.__class__.__name__ == "atyp"


@pytest.mark.parametrize(
    ("rtrn_f", "expected"),
    [
        (m.rtrn_valu, "rtrn_valu(_MvCtor)*_MvCtor"),
        (m.rtrn_rref, "rtrn_rref(_MvCtor)*_MvCtor"),
        (m.rtrn_cref, "rtrn_cref(_MvCtor)*_CpCtor"),
        (m.rtrn_mref, "rtrn_mref(_MvCtor)*_CpCtor"),
        (m.rtrn_cptr, "rtrn_cptr"),
        (m.rtrn_mptr, "rtrn_mptr"),
        (m.rtrn_shmp, "rtrn_shmp"),
        (m.rtrn_shcp, "rtrn_shcp"),
        (m.rtrn_uqmp, "rtrn_uqmp"),
        (m.rtrn_uqcp, "rtrn_uqcp"),
        (m.rtrn_udmp, "rtrn_udmp"),
        (m.rtrn_udcp, "rtrn_udcp"),
    ],
)
def test_cast(rtrn_f, expected):
    assert re.match(expected, m.get_mtxt(rtrn_f()))


@pytest.mark.parametrize(
    ("pass_f", "mtxt", "expected"),
    [
        (m.pass_valu, "Valu", "pass_valu:Valu(_MvCtor)*_CpCtor"),
        (m.pass_cref, "Cref", "pass_cref:Cref(_MvCtor)*_MvCtor"),
        (m.pass_mref, "Mref", "pass_mref:Mref(_MvCtor)*_MvCtor"),
        (m.pass_cptr, "Cptr", "pass_cptr:Cptr(_MvCtor)*_MvCtor"),
        (m.pass_mptr, "Mptr", "pass_mptr:Mptr(_MvCtor)*_MvCtor"),
        (m.pass_shmp, "Shmp", "pass_shmp:Shmp(_MvCtor)*_MvCtor"),
        (m.pass_shcp, "Shcp", "pass_shcp:Shcp(_MvCtor)*_MvCtor"),
        (m.pass_uqmp, "Uqmp", "pass_uqmp:Uqmp(_MvCtor)*_MvCtor"),
        (m.pass_uqcp, "Uqcp", "pass_uqcp:Uqcp(_MvCtor)*_MvCtor"),
    ],
)
def test_load_with_mtxt(pass_f, mtxt, expected):
    assert re.match(expected, pass_f(m.atyp(mtxt)))


@pytest.mark.parametrize(
    ("pass_f", "rtrn_f", "expected"),
    [
        (m.pass_udmp, m.rtrn_udmp, "pass_udmp:rtrn_udmp"),
        (m.pass_udcp, m.rtrn_udcp, "pass_udcp:rtrn_udcp"),
    ],
)
def test_load_with_rtrn_f(pass_f, rtrn_f, expected):
    assert pass_f(rtrn_f()) == expected


@pytest.mark.parametrize(
    ("pass_f", "rtrn_f", "expected"),
    [
        (m.pass_uqmp, m.rtrn_uqmp, "pass_uqmp:rtrn_uqmp"),
        (m.pass_uqcp, m.rtrn_uqcp, "pass_uqcp:rtrn_uqcp"),
        (m.pass_udmp, m.rtrn_udmp, "pass_udmp:rtrn_udmp"),
        (m.pass_udcp, m.rtrn_udcp, "pass_udcp:rtrn_udcp"),
    ],
)
def test_pass_unique_ptr_disowns(pass_f, rtrn_f, expected):
    obj = rtrn_f()
    assert pass_f(obj) == expected
    with pytest.raises(ValueError) as exc_info:
        pass_f(obj)
    assert str(exc_info.value) == (
        "Missing value for wrapped C++ type: Python instance was disowned."
    )


@pytest.mark.parametrize(
    ("pass_f", "rtrn_f"),
    [
        (m.pass_uqmp, m.rtrn_uqmp),
        (m.pass_uqcp, m.rtrn_uqcp),
        (m.pass_udmp, m.rtrn_udmp),
        (m.pass_udcp, m.rtrn_udcp),
    ],
)
def test_cannot_disown_use_count_ne_1(pass_f, rtrn_f):
    obj = rtrn_f()
    stash = m.SharedPtrStash()
    stash.Add(obj)
    with pytest.raises(ValueError) as exc_info:
        pass_f(obj)
    assert str(exc_info.value) == (
        "Cannot disown use_count != 1 (loaded_as_unique_ptr)."
    )


def test_unique_ptr_roundtrip(num_round_trips=1000):
    # Multiple roundtrips to stress-test instance registration/deregistration.
    recycled = m.atyp("passenger")
    for _ in range(num_round_trips):
        id_orig = id(recycled)
        recycled = m.unique_ptr_roundtrip(recycled)
        assert re.match("passenger(_MvCtor)*_MvCtor", m.get_mtxt(recycled))
        id_rtrn = id(recycled)
        # Ensure the returned object is a different Python instance.
        assert id_rtrn != id_orig
        id_orig = id_rtrn


# This currently fails, because a unique_ptr is always loaded by value
# due to pybind11/detail/smart_holder_type_casters.h:689
# I think, we need to provide more cast operators.
@pytest.mark.skip()
def test_unique_ptr_cref_roundtrip():
    orig = m.atyp("passenger")
    id_orig = id(orig)
    mtxt_orig = m.get_mtxt(orig)

    recycled = m.unique_ptr_cref_roundtrip(orig)
    assert m.get_mtxt(orig) == mtxt_orig
    assert m.get_mtxt(recycled) == mtxt_orig
    assert id(recycled) == id_orig


@pytest.mark.parametrize(
    ("pass_f", "rtrn_f", "moved_out", "moved_in"),
    [
        (m.uconsumer.pass_valu, m.uconsumer.rtrn_valu, True, True),
        (m.uconsumer.pass_rref, m.uconsumer.rtrn_valu, True, True),
        (m.uconsumer.pass_valu, m.uconsumer.rtrn_lref, True, False),
        (m.uconsumer.pass_valu, m.uconsumer.rtrn_cref, True, False),
    ],
)
def test_unique_ptr_consumer_roundtrip(pass_f, rtrn_f, moved_out, moved_in):
    c = m.uconsumer()
    assert not c.valid()
    recycled = m.atyp("passenger")
    mtxt_orig = m.get_mtxt(recycled)
    assert re.match("passenger_(MvCtor){1,2}", mtxt_orig)

    pass_f(c, recycled)
    if moved_out:
        with pytest.raises(ValueError) as excinfo:
            m.get_mtxt(recycled)
        assert "Python instance was disowned" in str(excinfo.value)

    recycled = rtrn_f(c)
    assert c.valid() != moved_in
    assert m.get_mtxt(recycled) == mtxt_orig


def test_py_type_handle_of_atyp():
    obj = m.py_type_handle_of_atyp()
    assert obj.__class__.__name__ == "pybind11_type"


def test_function_signatures(doc):
    assert (
        doc(m.args_shared_ptr)
        == "args_shared_ptr(arg0: m.class_sh_basic.atyp) -> m.class_sh_basic.atyp"
    )
    assert (
        doc(m.args_shared_ptr_const)
        == "args_shared_ptr_const(arg0: m.class_sh_basic.atyp) -> m.class_sh_basic.atyp"
    )
    assert (
        doc(m.args_unique_ptr)
        == "args_unique_ptr(arg0: m.class_sh_basic.atyp) -> m.class_sh_basic.atyp"
    )
    assert (
        doc(m.args_unique_ptr_const)
        == "args_unique_ptr_const(arg0: m.class_sh_basic.atyp) -> m.class_sh_basic.atyp"
    )
