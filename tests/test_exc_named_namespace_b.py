import named_namespace_b as m


def test_inspect():
    assert m.name == "NB"
