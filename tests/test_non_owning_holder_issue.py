from pybind11_tests import non_owning_holder_issue as m


def test_ValueHolder():
    vh = m.ValueHolder()
    assert vh.held_value is not None


def test_VectorValueHolder():
    vvh = m.VectorValueHolder()
    vvh0 = vvh.vec_val_hld[0]
    assert vvh0.held_value is not None
