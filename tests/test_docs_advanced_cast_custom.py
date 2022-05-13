from pybind11_tests import docs_advanced_cast_custom as m


def test_all():
    assert m.to_string(135) == "135"
    assert m.return_42() == 42
