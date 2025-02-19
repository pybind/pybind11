from __future__ import annotations

import pytest

from pybind11_tests import unnamed_namespace_a as m
from pybind11_tests import unnamed_namespace_b as mb

XFAIL_CONDITION = "not m.defined_WIN32_or__WIN32 and (m.defined___clang__ or m.defined__LIBCPP_VERSION)"
XFAIL_REASON = "Known issues: https://github.com/pybind/pybind11/pull/4319"


@pytest.mark.xfail(XFAIL_CONDITION, reason=XFAIL_REASON, strict=False)
@pytest.mark.parametrize(
    "any_struct", [m.unnamed_namespace_a_any_struct, mb.unnamed_namespace_b_any_struct]
)
def test_have_class_any_struct(any_struct):
    assert any_struct is not None


def test_have_at_least_one_class_any_struct():
    assert (
        m.unnamed_namespace_a_any_struct is not None
        or mb.unnamed_namespace_b_any_struct is not None
    )


@pytest.mark.xfail(XFAIL_CONDITION, reason=XFAIL_REASON, strict=True)
def test_have_both_class_any_struct():
    assert m.unnamed_namespace_a_any_struct is not None
    assert mb.unnamed_namespace_b_any_struct is not None
