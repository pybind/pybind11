from pybind11_tests import class_sh_getattr_issue as m


def test_drvd1_add_in_cpp_unique_ptr():
    f = m.Foo()
    assert f.bar() == 42
    assert f.something == "GetAttr: something"
