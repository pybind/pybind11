from pybind11_tests import class_sh_shared_ptr_copy_move as m


def test_shptr_copy():
    txt = m.test_ShPtr_copy()[0].get_history()
    assert txt == "FooShPtr_copy"


def test_smhld_copy():
    txt = m.test_SmHld_copy()[0].get_history()
    assert txt == "FooSmHld_copy"


def test_shptr_move():
    txt = m.test_ShPtr_move()[0].get_history()
    assert txt == "FooShPtr_move"


def test_smhld_move():
    txt = m.test_SmHld_move()[0].get_history()
    assert txt == "FooSmHld_move"


def _check_property(foo_typ, prop_typ, policy):
    o = m.Outer()
    name = f"{foo_typ}_{prop_typ}_{policy}"
    history = f"Foo{foo_typ}_Outer"
    f = getattr(o, name)
    assert f.get_history() == history
    # and try again to check that o did not get changed
    f = getattr(o, name)
    assert f.get_history() == history


def test_properties():
    for prop_typ in ("readonly", "readwrite", "property_readonly"):
        for foo_typ in ("ShPtr", "SmHld"):
            for policy in ("default", "copy", "move"):
                _check_property(foo_typ, prop_typ, policy)
