from __future__ import annotations

import exo_planet_c_api
import exo_planet_pybind11
import pytest

from pybind11_tests import cpp_transporter as home_planet


def NOtest_call_cpp_transporter_success():
    t_h = home_planet.Traveler("home")
    cap = t_h.__cpp_transporter__(
        home_planet.PYBIND11_PLATFORM_ABI_ID,
        home_planet.typeid_Traveler_name,
        "raw_pointer_ephemeral",
    )
    assert cap.__class__.__name__ == "PyCapsule"


def NOtest_call_cpp_transporter_platform_abi_id_mismatch():
    t_h = home_planet.Traveler("home")
    cap = t_h.__cpp_transporter__(
        home_planet.PYBIND11_PLATFORM_ABI_ID + "MISMATCH",
        home_planet.typeid_Traveler_name,
        "raw_pointer_ephemeral",
    )
    assert cap is None
    diag = t_h.__cpp_transporter__(
        home_planet.PYBIND11_PLATFORM_ABI_ID + "MISMATCH",
        home_planet.typeid_Traveler_name,
        "query_mismatch",
    )
    assert diag == "pybind11_platform_abi_id_mismatch"


def NOtest_call_cpp_transporter_type_id_name_mismatch():
    t_h = home_planet.Traveler("home")
    cap = t_h.__cpp_transporter__(
        home_planet.PYBIND11_PLATFORM_ABI_ID,
        home_planet.typeid_Traveler_name + "MISMATCH",
        "raw_pointer_ephemeral",
    )
    assert cap is None
    diag = t_h.__cpp_transporter__(
        home_planet.PYBIND11_PLATFORM_ABI_ID,
        home_planet.typeid_Traveler_name + "MISMATCH",
        "query_mismatch",
    )
    assert diag == "cpp_typeid_name_mismatch"


def test_home_only_basic():
    t_h = home_planet.Traveler("home")
    assert t_h.luggage == "home"
    assert home_planet.get_luggage(t_h) == "home"


def test_home_only_premium():
    p_h = home_planet.PremiumTraveler("home", 2)
    assert p_h.luggage == "home"
    assert home_planet.get_luggage(p_h) == "home"
    assert home_planet.get_points(p_h) == 2


def test_exo_only_basic():
    t_e = exo_planet_pybind11.Traveler("exo")
    assert t_e.luggage == "exo"
    assert exo_planet_pybind11.get_luggage(t_e) == "exo"


def test_exo_only_premium():
    p_e = exo_planet_pybind11.PremiumTraveler("exo", 3)
    assert p_e.luggage == "exo"
    assert exo_planet_pybind11.get_luggage(p_e) == "exo"
    assert exo_planet_pybind11.get_points(p_e) == 3


def test_home_passed_to_exo_basic():
    t_h = home_planet.Traveler("home")
    assert exo_planet_pybind11.get_luggage(t_h) == "home"


def test_exo_passed_to_home_basic():
    t_e = exo_planet_pybind11.Traveler("exo")
    assert home_planet.get_luggage(t_e) == "exo"


def test_home_passed_to_exo_premium():
    p_h = home_planet.PremiumTraveler("home", 2)
    assert exo_planet_pybind11.get_luggage(p_h) == "home"
    assert exo_planet_pybind11.get_points(p_h) == 2


def test_exo_passed_to_home_premium():
    p_e = exo_planet_pybind11.PremiumTraveler("exo", 3)
    assert home_planet.get_luggage(p_e) == "exo"
    assert home_planet.get_points(p_e) == 3


@pytest.mark.parametrize(
    "traveler_type", [home_planet.Traveler, exo_planet_pybind11.Traveler]
)
def test_exo_planet_c_api_traveler(traveler_type):
    t = traveler_type("socks")
    assert exo_planet_c_api.GetLuggage(t) == "socks"


@pytest.mark.parametrize(
    "premium_traveler_type",
    [home_planet.PremiumTraveler, exo_planet_pybind11.PremiumTraveler],
)
def test_exo_planet_c_api_premium_traveler(premium_traveler_type):
    pt = premium_traveler_type("gucci", 5)
    assert exo_planet_c_api.GetLuggage(pt) == "gucci"
    assert exo_planet_c_api.GetPoints(pt) == 5
