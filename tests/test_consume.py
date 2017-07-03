import pytest


def test_consume_argument(capture):
    from pybind11_tests import Box, Filter

    with capture:
        filt = Filter(4)
    assert capture == "Filter created."
    with capture:
        box_1 = Box(1)
        box_8 = Box(8)
    assert capture == """
        Box created.
        Box created.
    """

    assert Box.get_num_boxes() == 2

    with capture:
        filt.process(box_1)  # box_1 is not big enough, but process() leaks it
    assert capture == "Box is processed by Filter."

    assert Box.get_num_boxes() == 2

    with capture:
        filt.process(box_8)  # box_8 is destroyed by process() of filt
    assert capture == """
        Box is processed by Filter.
        Box destroyed.
    """

    assert Box.get_num_boxes() == 1  # box_1 still exists somehow, but we can't access it

    with capture:
        del filt
        del box_1
        del box_8
        pytest.gc_collect()
    assert capture == "Filter destroyed."

    assert Box.get_num_boxes() == 1  # 1 box is leaked, and we can't do anything
