from pybind11_tests import gil_scoped as m


def test_cross_module_gil():
    """Makes sure that the GIL can be acquired by another module from a GIL-released state."""
    m.test_cross_module_gil()  # Should not raise a SIGSEGV
