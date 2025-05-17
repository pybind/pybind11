from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from conftest import SanitizedString

from pybind11_tests import docs_advanced_cast_custom as m


def assert_negate_function(
    input_sequence: Sequence[float],
    target: tuple[float, float],
) -> None:
    output = m.negate(input_sequence)
    assert isinstance(output, tuple)
    assert len(output) == 2
    assert isinstance(output[0], float)
    assert isinstance(output[1], float)
    assert output == target


def test_negate(doc: SanitizedString) -> None:
    assert (
        doc(m.negate)
        == "negate(arg0: collections.abc.Sequence[float]) -> tuple[float, float]"
    )
    assert_negate_function([1.0, -1.0], (-1.0, 1.0))
    assert_negate_function((1.0, -1.0), (-1.0, 1.0))
    assert_negate_function([1, -1], (-1.0, 1.0))
    assert_negate_function((1, -1), (-1.0, 1.0))


def test_docs() -> None:
    ###########################################################################
    # PLEASE UPDATE docs/advanced/cast/custom.rst IF ANY CHANGES ARE MADE HERE.
    ###########################################################################
    point1 = [1.0, -1.0]
    point2 = m.negate(point1)
    assert point2 == (-1.0, 1.0)
