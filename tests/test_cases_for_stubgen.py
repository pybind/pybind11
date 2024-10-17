import pytest

from pybind11_tests import cases_for_stubgen as m

TEST_CASES = {
    "m.basics.answer.__doc__": 'answer() -> int\n\nanswer docstring, with end quote"\n',
    "m.basics.sum.__doc__": "sum(arg0: int, arg1: int) -> int\n\nmultiline docstring test, edge case quotes \"\"\"'''\n",
    "m.basics.midpoint.__doc__": "midpoint(left: float, right: float) -> float\n",
    "m.basics.weighted_midpoint.__doc__": "weighted_midpoint(left: float, right: float, alpha: float = 0.5) -> float\n",
    "m.basics.Point.__init__.__doc__": "__init__(*args, **kwargs)\nOverloaded function.\n\n1. __init__(self: pybind11_tests.cases_for_stubgen.basics.Point) -> None\n\n2. __init__(self: pybind11_tests.cases_for_stubgen.basics.Point, x: float, y: float) -> None\n",
    "m.basics.Point.distance_to.__doc__": "distance_to(*args, **kwargs)\nOverloaded function.\n\n1. distance_to(self: pybind11_tests.cases_for_stubgen.basics.Point, x: float, y: float) -> float\n\n2. distance_to(self: pybind11_tests.cases_for_stubgen.basics.Point, other: pybind11_tests.cases_for_stubgen.basics.Point) -> float\n",
    "m.basics.Point.length_unit.__doc__": "Members:\n\n  mm\n\n  pixel\n\n  inch",
    "m.basics.Point.angle_unit.__doc__": "Members:\n\n  radian\n\n  degree",
    "m.pass_user_type.__doc__": "pass_user_type(arg0: `test_cases_for_stubgen::UserType`) -> None\n",
    "m.return_user_type.__doc__": "return_user_type() -> `test_cases_for_stubgen::UserType`\n",
    "m.MapIntUserType.keys.__doc__": "keys(self: pybind11_tests.cases_for_stubgen.MapIntUserType) -> pybind11_tests.cases_for_stubgen.KeysView\n",
    "m.MapIntUserType.values.__doc__": "values(self: pybind11_tests.cases_for_stubgen.MapIntUserType) -> pybind11_tests.cases_for_stubgen.ValuesView\n",
    "m.MapIntUserType.items.__doc__": "items(self: pybind11_tests.cases_for_stubgen.MapIntUserType) -> pybind11_tests.cases_for_stubgen.ItemsView\n",
    "m.MapUserTypeInt.keys.__doc__": "keys(self: pybind11_tests.cases_for_stubgen.MapUserTypeInt) -> pybind11_tests.cases_for_stubgen.KeysView\n",
    "m.MapUserTypeInt.values.__doc__": "values(self: pybind11_tests.cases_for_stubgen.MapUserTypeInt) -> pybind11_tests.cases_for_stubgen.ValuesView\n",
    "m.MapUserTypeInt.items.__doc__": "items(self: pybind11_tests.cases_for_stubgen.MapUserTypeInt) -> pybind11_tests.cases_for_stubgen.ItemsView\n",
    "m.MapFloatUserType.keys.__doc__": "keys(self: pybind11_tests.cases_for_stubgen.MapFloatUserType) -> Iterator[float]\n",
    "m.MapFloatUserType.values.__doc__": "values(self: pybind11_tests.cases_for_stubgen.MapFloatUserType) -> Iterator[`test_cases_for_stubgen::UserType`]\n",
    "m.MapFloatUserType.__iter__.__doc__": "__iter__(self: pybind11_tests.cases_for_stubgen.MapFloatUserType) -> Iterator[tuple[float, `test_cases_for_stubgen::UserType`]]\n",
    "m.MapUserTypeFloat.keys.__doc__": "keys(self: pybind11_tests.cases_for_stubgen.MapUserTypeFloat) -> Iterator[`test_cases_for_stubgen::UserType`]\n",
    "m.MapUserTypeFloat.values.__doc__": "values(self: pybind11_tests.cases_for_stubgen.MapUserTypeFloat) -> Iterator[float]\n",
    "m.MapUserTypeFloat.__iter__.__doc__": "__iter__(self: pybind11_tests.cases_for_stubgen.MapUserTypeFloat) -> Iterator[tuple[`test_cases_for_stubgen::UserType`, float]]\n",
    "m.pass_std_array_int_2.__doc__": "pass_std_array_int_2(arg0: Annotated[list[int], FixedSize(2)]) -> None\n",
    "m.return_std_array_int_3.__doc__": "return_std_array_int_3() -> Annotated[list[int], FixedSize(3)]\n",
    "m.nested_case_01a.__doc__": "nested_case_01a(arg0: list[Annotated[list[int], FixedSize(2)]]) -> None\n",
    "m.nested_case_02a.__doc__": "nested_case_02a(arg0: list[`test_cases_for_stubgen::UserType`]) -> None\n",
    "m.nested_case_03a.__doc__": "nested_case_03a(arg0: dict[Annotated[list[int], FixedSize(2)], `test_cases_for_stubgen::UserType`]) -> None\n",
    "m.nested_case_04a.__doc__": "nested_case_04a(arg0: dict[list[Annotated[list[int], FixedSize(2)]], list[`test_cases_for_stubgen::UserType`]]) -> None\n",
    "m.nested_case_05a.__doc__": "nested_case_05a(arg0: list[dict[list[Annotated[list[int], FixedSize(2)]], list[`test_cases_for_stubgen::UserType`]]]) -> None\n",
    "m.nested_case_06a.__doc__": "nested_case_06a(arg0: dict[dict[list[Annotated[list[int], FixedSize(2)]], list[`test_cases_for_stubgen::UserType`]], list[dict[list[Annotated[list[int], FixedSize(2)]], list[`test_cases_for_stubgen::UserType`]]]]) -> None\n",
}


@pytest.mark.parametrize("test_case", TEST_CASES.keys())
def test_docstring(test_case):
    assert dir(m)  # Only direct use of m, to stop tooling from removing the import.
    # On some platforms the stl_binders module name prevails for KeysView, ValuesView, ItemsView.
    docstring = eval(test_case).replace(".stl_binders.", ".cases_for_stubgen.")
    expected = TEST_CASES[test_case]
    assert docstring == expected
