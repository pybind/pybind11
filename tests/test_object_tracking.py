import os

import pytest

from pybind11_tests import object_ as m


def test_objects():
    p = m.Pod()
    objects = []
    objects_ids = set()
    for _ in range(5):
        ref_internal_obj = p.beed()
        objects.append(ref_internal_obj)
        objects_ids.add(id(ref_internal_obj))
    for o1 in objects:
        for o2 in objects:
            assert o1 is o2
    assert len(objects_ids) == 1