import pytest

from pybind11_tests import cases_for_stubgen as m


@pytest.mark.parametrize(
    ("docstring", "expected"),
    [
        (
            m.pass_user_type.__doc__,
            'pass_user_type(arg0: Annotated[Any, "test_cases_for_stubgen::UserType"]) -> None\n',
        ),
        (
            m.return_user_type.__doc__,
            'return_user_type() -> Annotated[Any, "test_cases_for_stubgen::UserType"]\n',
        ),
        (
            m.MapIntUserType.keys.__doc__,
            "keys(self: pybind11_tests.cases_for_stubgen.MapIntUserType) -> pybind11_tests.cases_for_stubgen.KeysView[int]\n",
        ),
        (
            m.MapIntUserType.values.__doc__,
            'values(self: pybind11_tests.cases_for_stubgen.MapIntUserType) -> pybind11_tests.cases_for_stubgen.ValuesView[Annotated[Any, "test_cases_for_stubgen::UserType"]]\n',
        ),
        (
            m.MapIntUserType.items.__doc__,
            'items(self: pybind11_tests.cases_for_stubgen.MapIntUserType) -> pybind11_tests.cases_for_stubgen.ItemsView[int, Annotated[Any, "test_cases_for_stubgen::UserType"]]\n',
        ),
        (
            m.MapUserTypeInt.keys.__doc__,
            'keys(self: pybind11_tests.cases_for_stubgen.MapUserTypeInt) -> pybind11_tests.cases_for_stubgen.KeysView[Annotated[Any, "test_cases_for_stubgen::UserType"]]\n',
        ),
        (
            m.MapUserTypeInt.values.__doc__,
            "values(self: pybind11_tests.cases_for_stubgen.MapUserTypeInt) -> pybind11_tests.cases_for_stubgen.ValuesView[int]\n",
        ),
        (
            m.MapUserTypeInt.items.__doc__,
            'items(self: pybind11_tests.cases_for_stubgen.MapUserTypeInt) -> pybind11_tests.cases_for_stubgen.ItemsView[Annotated[Any, "test_cases_for_stubgen::UserType"], int]\n',
        ),
        (
            m.MapFloatUserType.keys.__doc__,
            "keys(self: pybind11_tests.cases_for_stubgen.MapFloatUserType) -> Iterator[float]\n",
        ),
        (
            m.MapFloatUserType.values.__doc__,
            'values(self: pybind11_tests.cases_for_stubgen.MapFloatUserType) -> Iterator[Annotated[Any, "test_cases_for_stubgen::UserType"]]\n',
        ),
        (
            m.MapFloatUserType.__iter__.__doc__,
            '__iter__(self: pybind11_tests.cases_for_stubgen.MapFloatUserType) -> Iterator[tuple[float, Annotated[Any, "test_cases_for_stubgen::UserType"]]]\n',
        ),
        (
            m.MapUserTypeFloat.keys.__doc__,
            'keys(self: pybind11_tests.cases_for_stubgen.MapUserTypeFloat) -> Iterator[Annotated[Any, "test_cases_for_stubgen::UserType"]]\n',
        ),
        (
            m.MapUserTypeFloat.values.__doc__,
            "values(self: pybind11_tests.cases_for_stubgen.MapUserTypeFloat) -> Iterator[float]\n",
        ),
        (
            m.MapUserTypeFloat.__iter__.__doc__,
            '__iter__(self: pybind11_tests.cases_for_stubgen.MapUserTypeFloat) -> Iterator[tuple[Annotated[Any, "test_cases_for_stubgen::UserType"], float]]\n',
        ),
        (
            m.pass_std_array_int_2.__doc__,
            "pass_std_array_int_2(arg0: Annotated[list[int], FixedSize(2)]) -> None\n",
        ),
        (
            m.return_std_array_int_3.__doc__,
            "return_std_array_int_3() -> Annotated[list[int], FixedSize(3)]\n",
        ),
    ],
)
def test_docstring(docstring, expected):
    assert docstring == expected
