from __future__ import annotations

import pytest

from pybind11_tests import class_sh_property_non_owning as m


@pytest.mark.parametrize("persistent_holder", [True, False])
@pytest.mark.parametrize(
    ("core_fld", "expected"),
    [
        ("core_fld_value_ro", (13, 24)),
        ("core_fld_value_rw", (13, 24)),
        ("core_fld_shared_ptr_ro", (14, 25)),
        ("core_fld_shared_ptr_rw", (14, 25)),
        ("core_fld_raw_ptr_ro", (14, 25)),
        ("core_fld_raw_ptr_rw", (14, 25)),
        ("core_fld_unique_ptr_rw", (15, 26)),
    ],
)
def test_core_fld_common(core_fld, expected, persistent_holder):
    if persistent_holder:
        h = m.DataFieldsHolder(2)
        for i, exp in enumerate(expected):
            c = getattr(h.vec_at(i), core_fld)
            assert c.int_value == exp
    else:
        for i, exp in enumerate(expected):
            c = getattr(m.DataFieldsHolder(2).vec_at(i), core_fld)
            assert c.int_value == exp
