import pytest

from pybind11_tests import cases_for_stubgen as m
from pybind11_tests import stl as test_stl


@pytest.mark.parametrize(
    ("docstring", "expected"),
    [
        (
            m.pass_user_type.__doc__,
            'pass_user_type(arg0: Annotated[Any, "test_cases_for_stubgen::user_type"]) -> None\n',
        ),
        (
            m.return_user_type.__doc__,
            'return_user_type() -> Annotated[Any, "test_cases_for_stubgen::user_type"]\n',
        ),
        (
            m.MapIntUserType.keys.__doc__,
            "keys(self: pybind11_tests.cases_for_stubgen.MapIntUserType) -> pybind11_tests.cases_for_stubgen.KeysView[int]\n",
        ),
        (
            m.MapIntUserType.values.__doc__,
            "values(self: pybind11_tests.cases_for_stubgen.MapIntUserType) -> pybind11_tests.cases_for_stubgen.ValuesView[test_cases_for_stubgen::user_type]\n",
        ),
        (
            m.MapIntUserType.items.__doc__,
            "items(self: pybind11_tests.cases_for_stubgen.MapIntUserType) -> pybind11_tests.cases_for_stubgen.ItemsView[int, test_cases_for_stubgen::user_type]\n",
        ),
        (
            m.MapUserTypeInt.keys.__doc__,
            "keys(self: pybind11_tests.cases_for_stubgen.MapUserTypeInt) -> pybind11_tests.cases_for_stubgen.KeysView[test_cases_for_stubgen::user_type]\n",
        ),
        (
            m.MapUserTypeInt.values.__doc__,
            "values(self: pybind11_tests.cases_for_stubgen.MapUserTypeInt) -> pybind11_tests.cases_for_stubgen.ValuesView[int]\n",
        ),
        (
            m.MapUserTypeInt.items.__doc__,
            "items(self: pybind11_tests.cases_for_stubgen.MapUserTypeInt) -> pybind11_tests.cases_for_stubgen.ItemsView[test_cases_for_stubgen::user_type, int]\n",
        ),
        (
            m.MapFloatUserType.keys.__doc__,
            "keys(self: pybind11_tests.cases_for_stubgen.MapFloatUserType) -> Iterator[float]\n",
        ),
        (
            m.MapFloatUserType.values.__doc__,
            'values(self: pybind11_tests.cases_for_stubgen.MapFloatUserType) -> Iterator[Annotated[Any, "test_cases_for_stubgen::user_type"]]\n',
        ),
        (
            m.MapFloatUserType.__iter__.__doc__,
            '__iter__(self: pybind11_tests.cases_for_stubgen.MapFloatUserType) -> Iterator[tuple[float, Annotated[Any, "test_cases_for_stubgen::user_type"]]]\n',
        ),
        (
            m.MapUserTypeFloat.keys.__doc__,
            'keys(self: pybind11_tests.cases_for_stubgen.MapUserTypeFloat) -> Iterator[Annotated[Any, "test_cases_for_stubgen::user_type"]]\n',
        ),
        (
            m.MapUserTypeFloat.values.__doc__,
            "values(self: pybind11_tests.cases_for_stubgen.MapUserTypeFloat) -> Iterator[float]\n",
        ),
        (
            m.MapUserTypeFloat.__iter__.__doc__,
            '__iter__(self: pybind11_tests.cases_for_stubgen.MapUserTypeFloat) -> Iterator[tuple[Annotated[Any, "test_cases_for_stubgen::user_type"], float]]\n',
        ),
        (
            test_stl.cast_array.__doc__,
            "cast_array() -> Annotated[list[int], FixedSize(2)]\n",
        ),
        (
            test_stl.load_array.__doc__,
            "load_array(arg0: Annotated[list[int], FixedSize(2)]) -> bool\n",
        ),
    ],
)
def test_docstring(docstring, expected):
    assert docstring == expected
