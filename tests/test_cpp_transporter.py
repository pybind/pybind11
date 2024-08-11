from __future__ import annotations

import exo_planet

from pybind11_tests import cpp_transporter as home_planet


def test_home_only():
    t_h = home_planet.Traveler("home")
    assert t_h.luggage == "home"
    assert home_planet.get_luggage(t_h) == "home"


def test_exo_only():
    t_e = exo_planet.Traveler("exo")
    assert t_e.luggage == "exo"
    assert exo_planet.get_luggage(t_e) == "exo"


def test_home_passed_to_exo():
    t_h = home_planet.Traveler("home")
    assert exo_planet.get_luggage(t_h) == "home"


def test_exo_passed_to_home():
    t_e = exo_planet.Traveler("exo")
    assert home_planet.get_luggage(t_e) == "exo"


def test_call_cpp_transporter():
    t_h = home_planet.Traveler("home")
    assert (
        t_h.__cpp_transporter__(
            "cpp_abi_code", "cpp_typeid_name", "raw_pointer_ephemeral"
        )
        is not None
    )
