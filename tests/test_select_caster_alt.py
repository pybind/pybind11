from pybind11_tests import select_caster_alt as m


def test_mock_casters():
    assert m.have_a_ns_num() == 121
